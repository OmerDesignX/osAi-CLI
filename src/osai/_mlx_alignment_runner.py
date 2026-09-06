"""Gradient alignment runner executed behind the offline network guard."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def run(config_path: str | Path) -> int:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    from .alignment import reward_advantages

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model, tokenizer = load(config["model"], trust_remote_code=False)
    model.freeze()
    load_adapters(model, config["adapter"])
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if not trainable or any(
        name.rsplit(".", 1)[-1] not in {"lora_a", "lora_b", "m"}
        for name in trainable
    ):
        raise ValueError("alignment must expose only LoRA/DoRA adapter tensors")
    model.train()
    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()
    records = config["examples"]
    if not records:
        raise ValueError("alignment dataset is empty")

    sequences = [_prepare_record(tokenizer, item, config["max_seq_length"]) for item in records]
    references = []
    model.eval()
    for sequence in sequences:
        chosen = float(_sequence_logprob(mx, nn, model, *sequence[0]).item())
        rejected = (
            float(_sequence_logprob(mx, nn, model, *sequence[1]).item())
            if sequence[1] is not None
            else 0.0
        )
        references.append((chosen, rejected))
    model.train()

    optimizer_name = config["optimizer"]
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(learning_rate=float(config["learning_rate"]))
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(learning_rate=float(config["learning_rate"]))
    else:
        raise ValueError(f"unsupported MLX alignment optimizer: {optimizer_name}")
    method = config["alignment_type"]
    advantages = None
    groups: list[list[int]] = []
    if method in {"reinforce", "rloo", "grpo"}:
        advantages = reward_advantages(
            method,
            [float(item["reward"]) for item in records],
            [item["prompt"] for item in records],
        )
    if method in {"rloo", "grpo"}:
        grouped: dict[str, list[int]] = {}
        for index, item in enumerate(records):
            grouped.setdefault(item["prompt"], []).append(index)
        groups = list(grouped.values())

    def record_objective(current_model, index):
        pair = sequences[index]
        policy_chosen = _sequence_logprob(mx, nn, current_model, *pair[0])
        policy_rejected = (
            _sequence_logprob(mx, nn, current_model, *pair[1])
            if pair[1] is not None
            else mx.array(0.0)
        )
        old_chosen, old_rejected = references[index]
        item = records[index]
        configured_old = item.get("old_logprob")
        reward = (
            advantages[index]
            if advantages is not None
            else float(1.0 if item.get("reward") is None else item["reward"])
        )
        return _objective(
            mx,
            method,
            policy_chosen,
            policy_rejected,
            mx.array(old_chosen if configured_old is None else configured_old),
            mx.array(old_rejected),
            beta=float(config["beta"]),
            gamma=float(config["gamma"]),
            clip=float(config["ppo_clip"]),
            reward=reward,
            binary_reward=item.get("reward") is not None,
        )

    def objective(current_model, indices):
        total = mx.array(0.0)
        for index in indices:
            total = total + record_objective(current_model, index)
        return total / len(indices)

    value_and_grad = nn.value_and_grad(model, objective)
    losses: list[float] = []
    initial_loss: float | None = None
    local_batch = max(1, int(config["batch_size"]) // world_size)
    for step in range(int(config["iterations"])):
        if groups:
            indices = groups[(step * world_size + rank) % len(groups)]
        else:
            start = step * local_batch * world_size + rank * local_batch
            indices = [(start + offset) % len(records) for offset in range(local_batch)]
        loss, gradients = value_and_grad(model, indices)
        mx.eval(loss)
        pre_update = float(loss.item())
        if not math.isfinite(pre_update):
            raise ValueError(f"alignment produced a non-finite pre-update loss: {pre_update}")
        if initial_loss is None:
            initial_loss = pre_update
        gradients = average_gradients(gradients)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, max_norm=1.0)
        mx.eval(gradient_norm)
        norm = float(gradient_norm.item())
        if not math.isfinite(norm):
            raise ValueError(
                "alignment produced a non-finite gradient norm "
                f"(loss={pre_update}, gradient_norm={norm})"
            )
        optimizer.update(model, gradients)
        post_loss = objective(model, indices)
        mx.eval(model.parameters(), optimizer.state, post_loss, gradient_norm)
        value = float(post_loss.item())
        if not math.isfinite(value):
            raise ValueError(
                "alignment produced a non-finite post-update loss "
                f"(loss={pre_update}, gradient_norm={norm}, post_loss={value})"
            )
        losses.append(value)
        if rank == 0:
            print(f"alignment_step={step + 1} loss={value:.8f}", flush=True)

    output = Path(config["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(output / "adapters.safetensors"), weights)
        metadata = json.loads(
            (Path(config["adapter"]) / "adapter_config.json").read_text(encoding="utf-8")
        )
        metadata["alignment"] = {
            "type": method,
            "iterations": int(config["iterations"]),
            "beta": float(config["beta"]),
            "optimizer": optimizer_name,
        }
        (output / "adapter_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "alignment_result.json").write_text(
            json.dumps(
                {
                    "losses": losses,
                    "initial_loss": initial_loss,
                    "method": method,
                    "optimizer": optimizer_name,
                    "reward_groups": len(groups),
                    "trainable_tensors": len(trainable),
                    "trainable_parameters": sum(value.size for value in trainable.values()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    # Synchronize before non-zero ranks exit while rank zero writes the adapter.
    mx.eval(mx.distributed.all_sum(mx.array(1), stream=mx.cpu))
    return 0


def _prepare_record(tokenizer, item, max_length):
    chosen_text = item.get("chosen", item.get("response"))
    chosen = _tokenize_response(tokenizer, item["prompt"], chosen_text, max_length)
    rejected_text = item.get("rejected")
    rejected = (
        _tokenize_response(tokenizer, item["prompt"], rejected_text, max_length)
        if rejected_text is not None
        else None
    )
    return chosen, rejected


def _tokenize_response(tokenizer, prompt, response, max_length):
    prompt_messages = [{"role": "user", "content": prompt}]
    messages = [*prompt_messages, {"role": "assistant", "content": response}]
    try:
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, return_dict=False
        )
        tokens = tokenizer.apply_chat_template(messages, return_dict=False)
    except (AttributeError, TypeError, ValueError):
        prompt_tokens = tokenizer.encode(f"user: {prompt}\nassistant:")
        tokens = tokenizer.encode(f"user: {prompt}\nassistant: {response}")
    tokens = list(tokens)
    offset = min(len(prompt_tokens), max(1, len(tokens) - 1))
    if len(tokens) > max_length:
        # Retain the end of the prompt and the entire supervised response when possible.
        removed = len(tokens) - max_length
        tokens = tokens[removed:]
        offset = max(1, offset - removed)
    if len(tokens) < 2 or offset >= len(tokens):
        raise ValueError("alignment record has no response tokens after tokenization")
    return tokens, offset


def _sequence_logprob(mx, nn, model, tokens, response_offset):
    array = mx.array([tokens])
    logits = model(array[:, :-1])
    targets = array[:, 1:]
    loss = nn.losses.cross_entropy(logits, targets)
    mask = mx.arange(targets.shape[1]) >= max(0, response_offset - 1)
    return -(loss * mask).sum() / mx.maximum(mask.sum(), mx.array(1))


def _objective(
    mx, method, pc, pr, rc, rr, *, beta, gamma, clip, reward, binary_reward=False
):
    margin = pc - pr
    relative = margin - (rc - rr)
    if method == "dpo":
        return mx.logaddexp(mx.array(0.0), -beta * relative)
    if method == "ipo":
        return mx.square(relative - 1.0 / (2.0 * beta))
    if method == "simpo":
        return mx.logaddexp(mx.array(0.0), -(beta * margin - gamma))
    if method == "cpo":
        return mx.logaddexp(mx.array(0.0), -beta * margin) - beta * pc
    if method == "orpo":
        maximum_log_probability = math.log1p(-1e-4)

        def log_odds(log_probability):
            bounded = mx.minimum(log_probability, maximum_log_probability)
            return bounded - mx.log(-mx.expm1(bounded))

        odds = log_odds(pc) - log_odds(pr)
        return -pc + beta * mx.logaddexp(mx.array(0.0), -odds)
    if method == "kto":
        if binary_reward:
            delta = pc - rc
            signed_delta = delta if reward > 0.0 else -delta
            return abs(reward) * (1.0 - mx.sigmoid(beta * signed_delta))
        kl = mx.maximum(0.0, 0.5 * ((pc - rc) + (pr - rr)))
        desirable = 1.0 - mx.sigmoid(beta * (pc - rc - kl))
        undesirable = 1.0 - mx.sigmoid(beta * (kl - pr + rr))
        return 0.5 * (desirable + undesirable)
    if method == "ppo":
        ratio = mx.exp(mx.clip(pc - rc, -20.0, 20.0))
        clipped = mx.clip(ratio, 1.0 - clip, 1.0 + clip)
        return -mx.minimum(ratio * reward, clipped * reward)
    if method in {"reinforce", "rloo"}:
        return -reward * pc
    if method == "grpo":
        ratio = mx.exp(mx.clip(pc - rc, -20.0, 20.0))
        clipped = mx.clip(ratio, 1.0 - clip, 1.0 + clip)
        reference_log_ratio = mx.clip(rc - pc, -20.0, 20.0)
        reference_ratio = mx.exp(reference_log_ratio)
        kl = reference_ratio - reference_log_ratio - 1.0
        return -mx.minimum(ratio * reward, clipped * reward) + beta * kl
    raise ValueError(f"unsupported alignment type: {method}")


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))
