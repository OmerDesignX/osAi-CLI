# osAi CLI

**osAi** trains LoRA adapters for quantized MLX and GGUF language and
vision-language models, and keeps the quantized base frozen.
Training and inference are local after any selected model download completes.

osCode Models are supported by default and custom models can be added.

More about osCode Models: https://models.omerdesign.com/oscode-models/

Supported modes:

- MLX gradient LoRA on Apple silicon and Linux.
- Quantized MLX-VLM gradient LoRA with local image, video, and compatible
  audio inputs.
- llama.cpp gradient LoRA on GGUF models.
- Post-fine-tuning DPO, IPO, SimPO, ORPO, CPO, KTO, PPO, REINFORCE,
  RLOO, and GRPO alignment.

MLX and GGUF alignment use reverse-mode gradients by default. The bundled
llama.cpp has a native weighted sequence-backward extension that accumulates a
preference pair into one LoRA update. Both engines keep the quantized base
frozen and update only LoRA adapter tensors. PPO, REINFORCE, RLOO, and GRPO
generate fresh answers from the fine-tuned local policy by default. osAi scores
those answers against the local alignment references and then backpropagates the
selected objective. No rollout, critic, reward-model, telemetry, or internet
server is started. Pass `--no-live-rollouts` only for an already-scored static
dataset.

AdamW optimizer is used for MLX while SGD is used for LLama.cpp models.

Hybrid GGUF models may alternate attention and recurrent blocks. osAi resolves
LoRA targets from the projections present in the model instead of assuming the
last blocks share one topology. When a supervised row exceeds the selected
GGUF context, llama.cpp uses overlapping windows so every assistant token
remains supervised. Each window can see only its local context. Rows
containing only raw text use full-token language-model loss. MLX text training
uses the same one-token overlap between bounded windows, covering every
supervised answer token. A window sees only its local context; Auto can reduce
the window size and then batch size after a device-memory failure. The trainer
reports each attempt and the actual count of tokenized windows.
Assistant-supervised GGUF training keeps each tokenized record at its own length
in host memory and pads only the active training batch. Training stops computing
after the final supervised token in each record. The selected context is still
the maximum allowed length, and the longest record must fit the model and GPU
memory; shorter records do not reserve a full-width dataset row for the run.
Native progress counts the microbatches actually used by those records.
For full-dataset GGUF Auto training, calibration measures a short pilot and
scales its learning rate by the square root of pilot record-passes relative
to planned record-passes. Context and batch remain hardware choices; only
the update pace changes with the amount of data. A short pilot cannot
guarantee later heldout loss will decline, so monitor saved checkpoints.

## Hardware support

| System | Engines |
| --- | --- |
| macOS 12 Monterey or 13 Ventura | llama.cpp on Metal or CPU |
| macOS 14 Sonoma or newer on Apple silicon | MLX and llama.cpp on Metal or CPU |
| Windows 10 or 11 | llama.cpp on CUDA, Vulkan, or CPU |
| Debian 12 / Ubuntu 22.04 or newer | MLX and llama.cpp on CUDA or CPU; llama.cpp also supports Vulkan |

The selected llama.cpp accelerator is checked against the binary that was
actually compiled. The App downloads this CLI on first launch, and the setup
script compiles llama.cpp on that computer. It installs CMake and Ninja into
the private Python environment. On Windows it detects Microsoft C++ Build
Tools, CUDA, and Vulkan SDK installations without manual path settings. When
the Vulkan runtime is present but its SDK is missing, setup downloads a
SHA-256-verified SDK into its private build cache. When Microsoft C++
runtime DLLs are missing or outdated, setup verifies and installs Microsoft's
pinned x64 Redistributable before compiling Vulkan shaders. Windows may request
administrator approval for this prerequisite; setup never restarts the computer.
When Microsoft C++ Build Tools are absent it downloads a verified portable C++
toolchain for CPU and Vulkan builds. CUDA compilation requires Microsoft C++ Build Tools and a CUDA
Toolkit. Linux can also use distribution Vulkan
development packages and `glslc` without defining `VULKAN_SDK`. A failed GPU
build tries another available GPU backend and reports an error
if none can be built. CPU-only setup must be selected explicitly.

osCode V1 and V2 GGUF small models use the Qwen3.5 hybrid architecture. Since
llama.cpp cannot backpropagate through its recurrent delta-net operation, the
automatic profile trains the final block's `mlp.down_proj` LoRA only. The
remaining blocks still participate in the frozen forward pass. Longer
structured records are windowed without discarding assistant labels; the CLI
reports the number of windowed records.

Training publishes both `base-plus-adapter` and a standalone lossless deployment
bundle by default. In both MLX and GGUF bundles, fusion keeps the quantized base
files byte-for-byte unchanged and embeds the exact adapter residual. The fusion
path performs no model dequantization or requantization.

## Install

Supported Python versions are **3.10, 3.11, 3.12, and 3.13**. Python 3.9 and
older and Python 3.14+ are rejected. The setup script installs CMake and Ninja;
macOS and Linux still need their native C/C++ compiler. CUDA and Vulkan builds
require their SDK/toolkit. A failed GPU build tries another available GPU
backend and reports an error if none can be built. CPU-only setup must be
selected explicitly.

### Recommended: one-command setup

Run the same Python script from the downloaded project directory on every OS.

macOS or Linux:

```sh
python3 scripts/setup_osai.py
```

Windows PowerShell:

```powershell
py -3.13 scripts\setup_osai.py
```

The script:

1. Detects the OS, architecture, macOS version, CUDA toolkit, and Vulkan tools, including conventional Windows SDK locations.
2. Creates or reuses `.venv`.
3. Selects the correct file from `requirements/` and installs it.
4. Installs `dist/osai-0.1.4-py3-none-any.whl`, or builds from the local
   project if the wheel is absent.
5. Builds the bundled MLX and MLX-LM sources when the platform supports MLX.
6. Builds vendored llama.cpp on this computer for Metal, CUDA, Vulkan, or CPU.
7. Runs `osai doctor`.

| Detected host | Requirements selected | Bundled engines built |
| --- | --- | --- |
| Apple silicon, macOS 14+ | `requirements/requirements.txt` | MLX/MLX-LM and llama.cpp Metal |
| macOS 12-13 or Intel Mac | `requirements/requirements-llama.txt` | llama.cpp Metal or CPU |
| Linux with CUDA 12 | `requirements/requirements-linux-cuda12.txt` | MLX CUDA and llama.cpp CUDA |
| Linux with CUDA 13 | `requirements/requirements-linux-cuda13.txt` | MLX CUDA and llama.cpp CUDA |
| Linux without supported CUDA | `requirements/requirements.txt` | MLX CPU and llama.cpp Vulkan or CPU |
| Windows 10 or 11 | `requirements/requirements-llama.txt` | llama.cpp CUDA, Vulkan, or CPU |

### Activate and verify

After the script completes, activate the environment.

macOS or Linux:

```sh
. .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then confirm the installation and local model bundle:

```sh
osai doctor
osai models
osai verify-models
osai select --tier small --engine auto
```

### Install the wheel manually

To install the wheel yourself first:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install dist/osai-0.1.4-py3-none-any.whl
python scripts/setup_osai.py --current-environment
```

On Windows, create the environment with `py -3.13 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1`. The final setup command installs the
hardware-specific libraries, builds the bundled engines, and runs diagnostics.

## Train

`--data` accepts one JSON, JSONL, NDJSON, or Parquet file, or a folder of
such files. Every supported file in the folder and its subfolders is used.
Names beginning with `valid`, `validation`, or `dev` form the validation
split; names beginning with `test` form the test split; other data files train.
Known metadata files are ignored. Parquet is converted in bounded batches to
session-local JSONL before training; source files are not changed. UTF-8 and
UTF-8-with-BOM JSON files are accepted. Equivalent supervised layouts can be
mixed; osAi validates every row and writes canonical splits inside the session.

| Dataset layout | Accepted fields |
| --- | --- |
| Language modelling | `text` |
| Standard or conversational completion | `prompt` + `completion`, as strings or message lists |
| OpenAI chat | `messages` with text/content parts, tools, and tool calls |
| ShareGPT or dialogue | `conversations`, `conversation`, `dialog`, `dialogue`, `chat`, or `turns`; `role/content`, `from/value`, `speaker/text`, and `author/content` messages |
| Alpaca / Dolly | `instruction` + optional `input` or `context` + `output` or `response` |
| QA / translation | `question/answer`, SQuAD `context/question/answers`, `query/response`, translation dictionaries, `source/target`, or `src/tgt` |
| Common task pairs | `problem/solution`, `task/solution`, `request/response`, `document/summary`, `article/highlights`, `description/code`, or `input/output` |
| Role columns | Optional `system` plus `user` and `assistant` text columns |
| Preference data used for SFT | `prompt` + `chosen/rejected`; the chosen response is the supervised target |

Image, audio, and video fields and multimodal message parts are parsed and
validated before training. Relative paths resolve beside the selected dataset;
absolute paths, base64 values, and `data:` URIs are also accepted. Remote media
URLs are never fetched.

```json
{"messages":[{"role":"user","content":"What is shown?"},{"role":"assistant","content":"A blue book."}],"images":["media/book.jpg"]}
{"prompt":"Describe the motion.","completion":"The pen moves left.","videos":["media/pen.mp4"]}
{"prompt":"Transcribe this clip.","completion":"Hello world.","audio":["media/hello.wav"]}
```

Media can also appear inside OpenAI-style content parts such as `image_url`,
`input_image`, `video`, `input_video`, `audio`, and `input_audio`. osAi copies or
materializes every input into the private session and records the detected
modalities in its manifest.

MLX VLM training requires a complete local quantized checkpoint containing the
matching processor and media-tower weights. Video is enabled only for model
families with a working local video processor; audio currently supports
compatible Gemma 4, MiniCPM-o, Nemotron Nano Omni, and Phi-4 Multimodal MLX
checkpoints. Capability checks inspect actual tensors rather than trusting
reserved media token IDs.

For a GGUF VLM, place a matching quantized MLX VLM under the custom model's
`mlx/` folder and the GGUF plus its `mmproj*.gguf` under `gguf/`. MLX performs
the media-conditioned backward pass because llama.cpp does not expose a media
projector backward API; osAi exports the language LoRA to GGUF and preserves the
GGUF and projector bytes. No full-precision model is created.

List the official model tiers and any custom local models:

```sh
osai models
```

Selecting an official tier with `osai select` or `osai train` downloads only the
chosen MLX or GGUF variant when it is not already present. V2 is the default;
use `--model-version v1` for the original osCode release. V2 also offers
`xsmall`. The downloader reads the [osModels V1/V2 catalogs](https://models.omerdesign.com/oscode-models/),
fetches individual files and split shards, and verifies each file against that
version's published SHA-256 list. V2 GGUF vision projectors are included. An
interrupted download can resume. The model filenames remain unchanged.
V1 models now live under `osCode-Models/V1/`, alongside V2 under `V2/`.
Previously downloaded V1 models in the original `MLX/` or `GGUF/` folders
remain usable; the desktop app promotes them into `V1/` during an upgrade
while preserving the original paths for older sessions.

Use `--no-download-model` or set
`OSAI_OFFLINE=1` to require an already-downloaded model.

Train Small with automatic engine and accelerator selection:

```sh
osai train \
  --tier small \
  --engine auto \
  --auto-settings \
  --data /path/to/data
```

Run fine-tuning and then automatic alignment:

```sh
osai train \
  --tier small \
  --stage fine-tune-align \
  --data /path/to/fine-tuning-data \
  --alignment-data /path/to/preference-data
```

Alignment accepts standard or conversational `prompt/chosen/rejected` rows,
implicit chosen/rejected conversations with a shared prompt, common
`preferred/non_preferred` and `winner/loser` aliases, ranked
`response_j/response_k` rows, numeric `prompt/response/reward` rows, and KTO
`prompt/completion/label` feedback. `question`, `query`, `answer`, `output`,
`score`, and `value` aliases are normalized where unambiguous. By
default, the fine-tuned adapter generates fresh answers for every source prompt.
Pairwise objectives compare the generated answer with the local references;
reward objectives use a local token/sequence-similarity score derived from those
references. RLOO and GRPO generate at least two answers per prompt and compute
their group baselines locally. `--alignment-type auto` selects DPO for
preference rows and PPO for reward rows.

Every generated answer is saved in the session's `rollouts/train.jsonl`, and
`rollouts/manifest.json` records the source dataset hash, fine-tuned adapter
hash, generation settings, and `network_used: false`. “Online” RL here means
that the current policy creates fresh experience during the run; it does not
mean an internet connection.

When `--alignment-type` is omitted from an interactive `fine-tune-align` run
using preference pairs, the CLI asks whether to use ORPO before training starts.
Answering no selects DPO. Non-interactive runs select DPO and print how to opt
into ORPO. Passing any method explicitly skips the question.

### Alignment method guide

| Method | Local guidance | What it optimizes | Use it when | Main trade-off |
| --- | --- | --- | --- | --- |
| DPO | Pairwise preferences | Reference-relative chosen/rejected margin | You want the dependable pairwise default | Keeps a frozen reference and depends on `beta` |
| IPO | Pairwise preferences | A finite squared preference-margin target | DPO over-separates noisy preference pairs | Target and `beta` need care |
| SimPO | Pairwise preferences | Reference-free, length-normalized margin | Memory is tight or no reference term is wanted | Margin `gamma` needs tuning |
| ORPO | Pairwise preferences | Chosen-response likelihood plus an odds-ratio preference loss | Chosen-answer quality should remain explicit during alignment | Stronger combined objective can overfit tiny data |
| CPO | Pairwise preferences | Reference-free contrastive margin plus chosen likelihood | You want a compact preference objective without a reference model | More sensitive to chosen-data quality |
| KTO | Binary feedback | Desirable responses up, undesirable responses down relative to the frozen policy | Feedback is thumbs-up/down rather than paired | Label balance and `beta` matter |
| PPO | Preference or reward references | Clipped policy-ratio surrogate over freshly generated answers | You want conservative policy updates | Sensitive to local reward quality and clipping |
| REINFORCE | Preference or reward references | Reward-weighted generated-answer log-probability | You want the simplest low-memory policy-gradient objective | High gradient variance |
| RLOO | Preference or reward references | REINFORCE with a leave-one-out rollout baseline | Fresh samples per prompt can reduce variance | Needs at least two generated answers per prompt |
| GRPO | Preference or reward references | Group-standardized clipped updates with a reference penalty | Several fresh samples and no learned critic are wanted | Needs reward variation inside every prompt group |

The larger method family is:

```text
                                      LLM ALIGNMENT / POST-TRAINING
                                                    |
                                  LOCAL EXECUTION -- NO SERVERS
                                                    |
                                      FINE-TUNED CURRENT POLICY
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
              LIVE ROLLOUTS (DEFAULT)                              STATIC DATA (OPT-OUT)
                         |                                                     |
               Generate fresh answers                              Use existing pairs or
               for each local prompt                               pre-scored response rows
                         |                                                     |
                 Score or pair answers                              --no-live-rollouts
                using local references
                         |
              +----------+-----------------------------------------+
              |                    |                               |
       PAIRWISE PREFERENCES   BINARY FEEDBACK                POLICY GRADIENT
              |                    |                               |
             DPO                  KTO                         REINFORCE
             IPO                                                 RLOO
            SimPO                                                GRPO
             ORPO                                                 PPO
             CPO
```

Live rollouts and their scoring are local for both MLX and llama.cpp. osAi's
built-in scorer uses the supplied local references; it does not silently call a
hosted judge or reward service. PPO is implemented as its clipped policy
surrogate without a learned critic model.

## Multi-GPU: Metal/CUDA/Vulkan

For GGUF fine-tuning and alignment, `--multi-gpu auto` discovers compatible
Metal, CUDA, or Vulkan devices. One native trainer divides the model by layer
across those devices and applies one optimizer update per step. The adapter
keeps the selected rank. `--multi-gpu on` requires at least two GPUs. Model
weights are shared across their memory, but activations and the largest layer
must still fit on their assigned GPU. Row and tensor splitting are not yet
supported for GGUF backpropagation. A 3 GiB card can participate only when its
assigned layers and training graph fit; a large model or full context may still
exceed it. Calibration reports that limit rather than switching to CPU training.
On Vulkan and Metal systems, automatic GGUF training prefers discrete cards
over recognized integrated adapters. Intel Macs can use a Metal eGPU when
llama.cpp lists it; Apple silicon Macs do not support eGPUs. MLX uses local
NCCL data parallelism on multi-GPU Linux CUDA systems; select the GGUF engine
when model sharding is required.

Automatic settings run a bounded one-turn inference benchmark using the selected
model and accelerator. GGUF devices are probed together with layer splitting.
The largest passing `compact`, `balanced`, `performance`, or
`maximum` profile must also fit conservative host and reported free-GPU-memory
reserves for training. Benchmark results are cached for 12 hours per model,
device set, and hardware state. Dataset size never changes the profile. GGUF
profiles range from 256 to 2048 context tokens and use smaller native
microbatches at larger contexts. The training pilot checks backward memory on
each GPU and lowers its microbatch after an allocation failure. Long supervised
records can take much longer to train because every assistant token remains in a window. Weighted
alignment requires each full sequence to fit the selected context and fails
explicitly otherwise. The selected values are printed before training and
stored in the run manifest.
Training epochs and learning rate remain user
controlled because they affect training duration and quality rather than peak
memory. One epoch means one full pass over the training split on both MLX and
llama.cpp. A manual flag such as `--rank 8` overrides that one automatic value.

## Custom models

Place local models in this structure:

```text
models/custom/my-model/
├── mlx/       # complete quantized MLX LM or VLM checkpoint
└── gguf/      # GGUF shard set and optional mmproj*.gguf
```

Then run:

```sh
osai train \
  --custom my-model \
  --engine auto \
  --data /path/to/data
```

To continue a finished run, use its `outputs/merged-model/` folder as the
custom model. It already has the required `gguf/` or `mlx/` layout. The CLI
resumes its embedded LoRA adapter, then writes a new adapter and merged model
into the next session. Automatic settings retain the adapter's rank and target
shape so previous learning is preserved.

## Output

```text
sessions/2026-09-05_14-30-00_small-auto/
├── manifests/
├── logs/
├── outputs/checkpoint/
│   ├── adapter/               # one replaceable latest adapter
│   └── merged-model/          # reusable model with the latest adapter
├── outputs/base-plus-adapter/
└── outputs/merged-model/
```

During a run, GGUF saves the current LoRA roughly every five minutes at the
next safe step and at completion; alignment also saves after each native
update. MLX saves every `--save-every` updates and at completion. A request
written to `checkpoint.request` in the session folder
also saves at the next optimizer step; replace its text with a new unique
value for each request. Model-sharded GGUF training writes one adapter at a
safe optimizer step. The adapter and the
`merged-model/` bundle are replaced in place, so numbered checkpoint copies
do not accumulate. The bundle can be selected as a custom model for a later
run. A checkpoint contains model weights, not optimizer state, so starting a
new run from it does not reproduce the interrupted optimizer exactly.

Combined runs place the supervised stage below `stages/fine-tuning/` and place
the final aligned adapter and merged model in the parent session's `outputs/`.
For MLX, the standalone output keeps every quantized base tensor unchanged and
embeds the adapter in `osai_adapter/`. The bundled MLX loader applies it from
`osai_fusion.json` automatically. Before publication, osAi compares every
next-token logit with the original base-plus-adapter path and requires an exact
match.

For GGUF, the bundle keeps the original file or split shards and any multimodal
projector unchanged under `model/`, stores the exact adapter as
`osai_adapter.gguf`, and records them in `osai_fusion.json`. SHA-256 checks
require every copied base, projector, and adapter file to match its source. Raw
llama.cpp commands load the manifest's model with
`--lora osai_adapter.gguf`; no unified, dequantized, or requantized model is
created.

Each run receives a new local date-and-time folder automatically. Use
`--session-name NAME` to change its suffix or `--sessions-root PATH` to choose a
different sessions directory.

## License

osAi is Apache-2.0 licensed. Vendored projects and separately downloaded models
retain their own licenses; see `THIRD_PARTY_NOTICES.md`.

## CLI reference

Use `osai --help` or `osai COMMAND --help` at any time.

Global options:

| Option | Purpose |
| --- | --- |
| `-h`, `--help` | Show help. |
| `--version` | Print the installed osai version. |

Commands:

| Command | Purpose |
| --- | --- |
| `doctor` | Report the OS, hardware, installed engines, and accelerators. |
| `inspect MODEL` | Inspect one local MLX or GGUF model. |
| `train` | Train and publish a timestamped model session. |
| `models` | List bundled and custom local models. |
| `verify-models` | Verify downloaded official models without network access. |
| `check-sessions` | Validate published sessions. |
| `select` | Show which local model and engine would be selected. |
| `export-gguf` | Convert an MLX LoRA adapter to GGUF adapter format. |
| `build-llama` | Build the vendored llama.cpp tools in a writable native build directory. Managed App installations use a short per-install directory so CUDA compiler output paths stay within Windows limits. |
| `validate-gguf` | Load a GGUF base and adapter together. |
| `prove-learning` | Compare deterministic output before and after an adapter. |

### `train` options

Choose exactly one model source: `--config`, `--tier`, or `--custom`.
Fine-tuning needs `--data`; alignment needs `--alignment-data` and an existing
`--adapter`; `fine-tune-align` needs both datasets and passes its trained adapter
directly to alignment.

| Option | Purpose |
| --- | --- |
| `--config PATH` | Load training settings from a local TOML or JSON file. |
| `--tier xsmall\|small\|medium\|large` | Use an official osCode model tier, downloading the selected format if needed. xSmall requires V2. |
| `--custom NAME` | Use `models/custom/NAME/mlx` or `gguf`. |
| `--engine auto\|mlx\|llama.cpp` | Select the trainer. `auto` prefers MLX on Apple silicon and llama.cpp elsewhere. |
| `--accelerator auto\|metal\|mps\|cuda\|vulkan\|cpu` | Select compute. `auto` is GPU-first with CPU fallback. MPS is diagnostic only; use Metal on macOS. |
| `--stage fine-tuning\|alignment\|fine-tune-align` | Select one stage or run fine-tuning followed by alignment. Default: `fine-tuning`. |
| `--data PATH` | Directory containing `train.jsonl` and optional validation/test splits in any supported supervised schema. |
| `--alignment-data PATH` | Directory containing standard or conversational preference, binary-feedback, or scored-response `train.jsonl`. |
| `--alignment-type auto\|dpo\|ipo\|simpo\|orpo\|cpo\|kto\|ppo\|reinforce\|rloo\|grpo` | Alignment objective. Omit it in `fine-tune-align` with preference data to receive a yes/no ORPO prompt. Explicit `auto` chooses DPO for preference pairs and PPO for reward rows. |
| `--adapter PATH` | Input adapter file/directory for an alignment-only run. |
| `--alignment-iterations N` | Alignment optimizer updates. Default: `10`. |
| `--alignment-learning-rate NUMBER` | Alignment learning rate. Default: `1e-5` for MLX and `0.001` for GGUF. |
| `--alignment-beta NUMBER` | Preference loss temperature/weight. Default: `0.1`. |
| `--alignment-gamma NUMBER` | SimPO target margin. Default: `0.5`. |
| `--ppo-clip NUMBER` | PPO clipping width. Default: `0.2`. |
| `--live-rollouts`, `--no-live-rollouts` | Generate fresh answers from the fine-tuned local policy before alignment, or consume the supplied rows unchanged. Enabled by default. |
| `--rollouts-per-prompt N` | Fresh answers generated per source prompt. Default: `2`; RLOO and GRPO require at least `2`. |
| `--rollout-max-tokens N` | Maximum new tokens in each generated answer. Default: `32`. |
| `--rollout-temperature NUMBER` | Local rollout sampling temperature. Default: `0.8`; use `0` for deterministic generation. |
| `--rollout-top-p NUMBER` | Local nucleus-sampling probability. Default: `0.95`. |
| `--rollout-seed N` | Starting seed for reproducible local rollout sampling. Default: `0`. |
| `--sessions-root PATH` | Store sessions below this directory. Default: `sessions/`. |
| `--session-name NAME` | Replace the generated folder suffix while retaining its timestamp. |
| `--bundled-root PATH` | Override the `osCode-Models` directory. |
| `--model-version v1\|v2` | Choose the official osCode release. Default: V2; V1 remains available. |
| `--custom-root PATH` | Override the `models/custom` directory. |
| `--download-model`, `--no-download-model` | Download and verify a missing official tier, or require it to exist locally. Enabled by default. |
| `--multi-gpu auto\|on\|off` | Automatically use available devices, require multiple GPUs, or force one GPU. |
| `--device NAME` | llama.cpp device name; repeat to set device order. |
| `--split-mode none\|layer\|row\|tensor` | llama.cpp model split. Default: `layer`. |
| `--tensor-split LIST` | Comma-separated llama.cpp device proportions, such as `3,1`. |
| `--main-gpu N` | llama.cpp main GPU index. Default: `0`. |
| `--distributed-workers N` | MLX Linux CUDA/NCCL worker count. `0` chooses a safe count. |
| `--auto-settings`, `--no-auto-settings` | With Auto, scan the selected training files for the longest record first, check the model context limit, then fit the hardware and calibrate a short local sample before the full run. Auto is disabled by default. Explicit tuning flags override its choices. |
| `--epochs N` | Complete passes over the fine-tuning dataset on MLX or llama.cpp. Default: `1`. |
| `--iterations N` | Compatibility alias for `--epochs`. |
| `--batch-size N` | MLX training batch size. Default: `1`. |
| `--rank N` | LoRA rank. Default: `2`. |
| `--scale NUMBER` | LoRA scaling value. Default: `4`. |
| `--num-layers N` | Number of final model layers to adapt. Default: `1`. |
| `--max-seq-length N` | MLX or GGUF context tokens per training window. Longer supervised rows are windowed without dropping assistant labels; each window sees only local context. Auto selects a hardware-aware value. |
| `--full-content-context`, `--no-full-content-context` | Full is the default with `--auto-settings` and for `calibrate`. Scan every training record and use a context sized for the largest one. GGUF uses the model's native tokenizer; MLX uses its `tokenizer.json` when available. A model or device memory limit is reported without silently reducing the full context. Use `--no-full-content-context` to select overlapping windows. |
| `--calibrated-context N` | Reuse the required context returned by a completed Full calibration without repeating the token scan at training startup. Use only after verifying that the dataset and model have not changed; `--max-seq-length` must cover `N`. The osAi App performs that verification before passing this option. |
| `--image-size WIDTH HEIGHT` | Resize local images before VLM preprocessing. Omit it to use the model processor's native size. |
| `--video-fps NUMBER` | Frames sampled per second from local videos. Default: `2`. |
| `--video-max-frames N` | Maximum frames loaded from each local video. Default: `32`. |
| `--assistant-token-id N` | Assistant boundary token used for completion-only VLM loss when a processor cannot expose it automatically. |
| `--learning-rate NUMBER` | Override the backend learning rate. |
| `--dropout NUMBER` | MLX LoRA dropout in the range `[0, 1)`. Default: `0`. |
| `--seed N` | Fine-tuning random seed. Default: `0`. |
| `--gradient-accumulation-steps N` | MLX microbatches accumulated before each optimizer update. Default: `1`. |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | Recompute MLX activations during backward to reduce memory use, or retain them for speed. Enabled by default. |
| `--save-every N` | Replace the latest MLX adapter every N updates. Default: `10`. |
| `--steps-per-report N` | Report MLX training metrics every N updates. Default: `1`. |
| `--steps-per-eval N` | Evaluate the MLX adapter every N updates. Default: `10`. |
| `--val-batches N` | MLX validation batches; `-1` uses the complete validation split. Default: `1`. |
| `--mask-prompt`, `--no-mask-prompt` | Include or exclude prompt tokens from supervised loss. Masking is enabled by default. |
| `--optimizer auto\|sgd\|adamw` | Optimizer for fine-tuning and alignment. `auto` uses AdamW with MLX and SGD with llama.cpp. |
| `--gguf-batch-size N` | GGUF backpropagation microbatch size. Default: `8`. |
| `--gguf-threads N` | CPU threads used by GGUF training and fusion validation. Default: `2`. |
| `--target-module NAME` | Adapt a projection; repeat for several. Choices: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`. |
| `--strict-base-hash`, `--no-strict-base-hash` | Enable or disable full pre/post base-file hashes. Enabled by default. |
| `--merge`, `--no-merge` | Enable or disable the standalone lossless quantized-residual bundle. Enabled by default. |
| `--materialize-base`, `--no-materialize-base` | Copy/clone the base into the deployment folder or reference its local path. Enabled by default. |
| `--python PATH` | Python executable containing MLX, MLX LM, and MLX-VLM. |

### Other command options

| Command | Option | Purpose |
| --- | --- | --- |
| `doctor` | `--json` | Print the hardware report as JSON. |
| `inspect` | `MODEL` | Local model file or directory to inspect. |
| `inspect` | `--format mlx\|gguf` | Require a specific format instead of detecting it. |
| `models` | `--bundled-root PATH` | Override the official-model download directory. |
| `models` | `--model-version v1\|v2` | List the selected official release; default V2. |
| `models` | `--custom-root PATH` | Override the custom-model directory. |
| `models` | `--json` | Print the catalog as JSON. |
| `verify-models` | `--root PATH` | Downloaded-model directory to verify. Default: `osCode-Models/`. |
| `check-sessions` | `--root PATH` | Sessions directory to inspect. Default: `sessions/`. |
| `check-sessions` | `--require-completed` | Fail when the directory has no completed session. |
| `select` | `--tier xsmall\|small\|medium\|large` | Select an official tier. xSmall requires V2. Use this or `--custom`. |
| `select` | `--model-version v1\|v2` | Choose the official release; default V2. |
| `select` | `--custom NAME` | Select a custom model. Use this or `--tier`. |
| `select` | `--engine auto\|mlx\|llama.cpp` | Resolve for a specific engine. Default: `auto`. |
| `select` | `--bundled-root PATH` | Override the official-model download directory. |
| `select` | `--download-model`, `--no-download-model` | Download and verify a missing tier, or require it to exist locally. |
| `select` | `--custom-root PATH` | Override the custom-model directory. |
| `auto-devices` | `--accelerator auto\|metal\|cuda\|vulkan\|cpu` | List currently available native GPU devices without loading a model. |
| `auto-benchmark` | `--tier` or `--custom`, plus model and accelerator options | Run the local inference probe and print selected training settings as JSON. `--refresh` bypasses the cache. |
| `calibrate` | `--tier` or `--custom`, `--data`, plus model and accelerator options | Sample short excerpts across the selected files locally, probe a GGUF microbatch on every selected GPU with memory backoff, compare training and held-out loss, and test a faster learning rate when the measured decline is small. Print the measured rate and fitted settings as JSON. Pilot progress and native logs go to stderr so stdout stays machine-readable. Full training uses the original records. A falling pilot loss does not guarantee a monotonic full run. |
| `export-gguf` | `--adapter PATH` | Source MLX adapter directory. Required. |
| `export-gguf` | `--base PATH` | Matching GGUF base model. Required. |
| `export-gguf` | `--output PATH` | Destination GGUF adapter file. Required. |
| `export-gguf` | `--dtype f16\|f32` | GGUF adapter tensor type. Default: `f16`. |
| `build-llama` | `--log PATH` | Build log destination. Default: `build/llama-build.log`. |
| `build-llama` | `--jobs N` | Parallel compiler jobs. Default: build-tool automatic. |
| `build-llama` | `--accelerator auto\|metal\|mps\|cuda\|vulkan\|cpu` | Backend to compile. Default: `auto`; MPS is not a llama.cpp backend. |
| `build-llama` | `--no-cpu-fallback` | Fail instead of retrying a failed GPU build on CPU. |
| `validate-gguf` | `--base PATH` | Quantized GGUF base. Required. |
| `validate-gguf` | `--adapter PATH` | GGUF LoRA adapter. Required. |
| `validate-gguf` | `--log PATH` | Validation log destination. Required. |
| `validate-gguf` | `--prompt TEXT` | Validation prompt. Default: a short success request. |
| `validate-gguf` | `--tokens N` | Maximum generated tokens. Default: `8`. |
| `validate-gguf` | `--context N` | Inference context size. Default: `128`. |
| `validate-gguf` | `--accelerator auto\|metal\|mps\|cuda\|vulkan\|cpu` | Validation compute backend. Default: `auto`. |
| `prove-learning` | `--engine mlx\|llama.cpp` | Adapter engine. Required. |
| `prove-learning` | `--model PATH` | Quantized base model. Required. |
| `prove-learning` | `--adapter PATH` | Trained adapter. Required. |
| `prove-learning` | `--prompt TEXT` | Deterministic test prompt. Required. |
| `prove-learning` | `--expected TEXT` | Text that must appear only after adaptation. Required. |
| `prove-learning` | `--output PATH` | JSON proof destination. Required. |
| `prove-learning` | `--max-tokens N` | Maximum generated tokens per probe. Default: `16`. |
| `prove-learning` | `--context N` | Inference context size. Default: `128`. |
| `prove-learning` | `--python PATH` | Python executable containing MLX. |
| `prove-learning` | `--accelerator auto\|metal\|mps\|cuda\|vulkan\|cpu` | Probe compute backend. Default: `auto`. |

### Advanced command examples

Fine-tune a quantized VLM from local images:

```sh
osai train \
  --custom my-vlm \
  --engine mlx \
  --stage fine-tuning \
  --data /path/to/image-dataset \
  --auto-settings \
  --max-seq-length 2048
```

Fine-tune with explicit training controls:

```sh
osai train \
  --tier small \
  --engine auto \
  --accelerator auto \
  --stage fine-tuning \
  --data /path/to/fine-tuning-data \
  --no-auto-settings \
  --optimizer sgd \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --max-seq-length 512 \
  --learning-rate 0.0001 \
  --rank 8 \
  --scale 16 \
  --num-layers 8 \
  --dropout 0.05 \
  --seed 42 \
  --save-every 20 \
  --steps-per-report 2 \
  --steps-per-eval 20 \
  --val-batches 4 \
  --target-module self_attn.q_proj \
  --target-module self_attn.v_proj
```

Fine-tune and then align with local rollouts:

```sh
osai train \
  --tier small \
  --engine auto \
  --stage fine-tune-align \
  --data /path/to/fine-tuning-data \
  --alignment-data /path/to/preference-data \
  --alignment-type grpo \
  --auto-settings \
  --optimizer auto \
  --epochs 1 \
  --alignment-iterations 40 \
  --alignment-learning-rate 0.00001 \
  --alignment-beta 0.1 \
  --ppo-clip 0.2 \
  --live-rollouts \
  --rollouts-per-prompt 4 \
  --rollout-max-tokens 128 \
  --rollout-temperature 0.8 \
  --rollout-top-p 0.95 \
  --rollout-seed 42 \
  --multi-gpu auto
```
