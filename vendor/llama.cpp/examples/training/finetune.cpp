#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

static bool lora_param_filter(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(userdata);
    const std::string name(tensor->name);
    const std::string suffix_a = ".lora_a";
    const std::string suffix_b = ".lora_b";
    const auto ends_with = [&name](const std::string & suffix) {
        return name.size() >= suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    return ends_with(suffix_a) || ends_with(suffix_b);
}

static bool token_sequence_at(
        const std::vector<llama_token> & tokens,
        size_t                           offset,
        const std::vector<llama_token> & sequence) {
    return !sequence.empty() && offset + sequence.size() <= tokens.size() &&
            std::equal(sequence.begin(), sequence.end(), tokens.begin() + offset);
}

// Build one padded datapoint per structured record and mark every label outside
// an `assistant:` span with a negative sentinel. This avoids target leakage
// between repeated sliding windows and makes short supervised JSONL examples
// trainable without constructing a full-precision model copy.
static ggml_opt_dataset_t assistant_dataset_init(
        llama_context      * ctx,
        const std::string  & corpus,
        int64_t            & trained_labels,
        int64_t            & n_examples,
        std::vector<int64_t> & label_counts,
        std::vector<int64_t> & backward_batches) {
    const auto assistant = common_tokenize(ctx, "assistant:", false);
    const auto user      = common_tokenize(ctx, "user:",      false);
    if (assistant.empty() || user.empty()) {
        return nullptr;
    }

    const std::string separator = "\n<|osai_record_end|>\n";
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const llama_token eos = llama_vocab_eos(vocab);
    std::vector<std::vector<llama_token>> examples;
    std::vector<std::vector<bool>> train_masks;
    for (size_t begin = 0; begin <= corpus.size();) {
        const size_t end = corpus.find(separator, begin);
        std::string record = corpus.substr(begin, end == std::string::npos ? end : end - begin);
        while (!record.empty() && (record.back() == '\n' || record.back() == '\r')) {
            record.pop_back();
        }
        if (!record.empty()) {
            auto tokens = common_tokenize(ctx, record, true);
            if (eos != LLAMA_TOKEN_NULL && (tokens.empty() || tokens.back() != eos)) {
                tokens.push_back(eos);
            }
            if (tokens.size() > llama_n_ctx(ctx)) {
                LOG_ERR("structured training record has %zu tokens but context size is %u\n",
                        tokens.size(), llama_n_ctx(ctx));
                return nullptr;
            }
            std::vector<bool> train_token(tokens.size(), false);
            bool in_assistant = false;
            bool found_assistant = false;
            for (size_t i = 0; i < tokens.size();) {
                if (token_sequence_at(tokens, i, assistant)) {
                    found_assistant = true;
                    in_assistant = true;
                    i += assistant.size();
                    continue;
                }
                if (token_sequence_at(tokens, i, user)) {
                    in_assistant = false;
                    i += user.size();
                    continue;
                }
                train_token[i] = in_assistant;
                ++i;
            }
            if (!found_assistant) {
                LOG_ERR("structured training record has no assistant: span\n");
                return nullptr;
            }
            examples.push_back(std::move(tokens));
            train_masks.push_back(std::move(train_token));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + separator.size();
    }
    if (examples.empty()) {
        return nullptr;
    }

    const int64_t n_ctx = llama_n_ctx(ctx);
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, examples.size(), 1);
    auto * data   = (llama_token *) ggml_opt_dataset_data(dataset)->data;
    auto * labels = (llama_token *) ggml_opt_dataset_labels(dataset)->data;

    llama_token padding = llama_vocab_pad(vocab);
    if (padding == LLAMA_TOKEN_NULL) {
        padding = llama_vocab_eos(vocab);
    }
    if (padding == LLAMA_TOKEN_NULL) {
        padding = llama_vocab_bos(vocab);
    }
    if (padding == LLAMA_TOKEN_NULL) {
        padding = 0;
    }

    trained_labels = 0;
    n_examples = examples.size();
    label_counts.assign(n_examples, 0);
    backward_batches.assign(n_examples, 0);
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    for (int64_t idata = 0; idata < n_examples; ++idata) {
        std::fill(data + idata*n_ctx, data + (idata + 1)*n_ctx, padding);
        std::fill(labels + idata*n_ctx, labels + (idata + 1)*n_ctx, -1);
        std::copy(examples[idata].begin(), examples[idata].end(), data + idata*n_ctx);
        std::vector<bool> has_labels_by_ubatch(
                (examples[idata].size() + n_ubatch - 1)/n_ubatch, false);
        for (size_t target = 1; target < examples[idata].size(); ++target) {
            if (train_masks[idata][target]) {
                const size_t label_index = target - 1;
                labels[idata*n_ctx + label_index] = examples[idata][target];
                ++trained_labels;
                ++label_counts[idata];
                has_labels_by_ubatch[label_index/n_ubatch] = true;
            }
        }
        backward_batches[idata] = std::count(
                has_labels_by_ubatch.begin(), has_labels_by_ubatch.end(), true);
    }
    return dataset;
}

static bool parse_example_weights(
        const char * raw,
        size_t expected,
        std::vector<float> & weights) {
    if (raw == nullptr || *raw == '\0') {
        return false;
    }
    std::stringstream input(raw);
    std::string part;
    while (std::getline(input, part, ',')) {
        std::stringstream value_input(part);
        float value = 0.0f;
        char trailing = 0;
        if (!(value_input >> value) || (value_input >> trailing) || !std::isfinite(value)) {
            return false;
        }
        weights.push_back(value);
    }
    return weights.size() == expected;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.load_mode != LLAMA_LOAD_MODE_NONE) {
        LOG_INF("%s: forcing load_mode = none to enable writable pointers to the weights\n", __func__);
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    // load the model and apply lora adapter, if any
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const bool adapter_only = !llama_init->lora().empty();
    if (llama_init->lora().size() > 1) {
        LOG_ERR("%s: adapter training accepts exactly one --lora file\n", __func__);
        return 1;
    }
    if (adapter_only && params.lora_adapters.front().scale != 1.0f) {
        LOG_ERR("%s: adapter training requires the --lora scale to be 1.0\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    const char * mask_prompt = std::getenv("OSAI_MASK_PROMPT");
    const bool assistant_only = adapter_only && mask_prompt != nullptr && strcmp(mask_prompt, "1") == 0;
    const char * example_weights_env = std::getenv("OSAI_EXAMPLE_WEIGHTS");
    const bool weighted_alignment = assistant_only && example_weights_env != nullptr;
    std::vector<int64_t> label_counts;
    std::vector<int64_t> backward_batches;
    std::vector<float> example_weights;
    int64_t weighted_opt_period = 0;
    ggml_opt_dataset_t dataset = nullptr;
    if (assistant_only) {
        int64_t trained_labels = 0;
        int64_t n_examples = 0;
        dataset = assistant_dataset_init(
                ctx, params.prompt, trained_labels, n_examples,
                label_counts, backward_batches);
        if (dataset == nullptr || trained_labels == 0) {
            LOG_ERR("%s: could not construct an assistant-only dataset\n", __func__);
            return 1;
        }
        LOG_INF("assistant-only loss enabled for %" PRId64 " labels in %" PRId64 " examples\n",
                trained_labels, n_examples);
        if (weighted_alignment) {
            if (n_examples < 1 || n_examples > 2) {
                LOG_ERR("%s: native alignment accepts one reward sequence or one preference pair\n",
                        __func__);
                return 1;
            }
            if (!parse_example_weights(example_weights_env, n_examples, example_weights)) {
                LOG_ERR("%s: invalid OSAI_EXAMPLE_WEIGHTS for %" PRId64 " examples\n",
                        __func__, n_examples);
                return 1;
            }
            if (params.lr.epochs != 1 || params.val_split != 0.0f) {
                LOG_ERR("%s: weighted alignment requires exactly one epoch and val-split 0\n",
                        __func__);
                return 1;
            }
            weighted_opt_period = std::accumulate(
                    backward_batches.begin(), backward_batches.end(), int64_t(0));
            if (weighted_opt_period < 1 ||
                    weighted_opt_period > std::numeric_limits<int32_t>::max()) {
                LOG_ERR("%s: invalid native alignment accumulation period: %" PRId64 "\n",
                        __func__, weighted_opt_period);
                return 1;
            }
            LOG_INF("native alignment gradient enabled for %" PRId64 " sequences\n", n_examples);
        }
    } else {
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
        if (tokens.size() <= llama_n_ctx(ctx) + 1) {
            LOG_ERR("%s: training corpus has %zu tokens but must have more than context size %u\n",
                    __func__, tokens.size(), llama_n_ctx(ctx));
            return 1;
        }
        dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);
    }

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/adapter_only ? lora_param_filter : llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
        /*opt_period      =*/weighted_alignment ? (int32_t) weighted_opt_period : 0,
    };
    if (weighted_alignment && lopt_params.opt_period < 1) {
        LOG_ERR("%s: weighted alignment has no backward microbatches\n", __func__);
        return 1;
    }
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();
    const char * eval_only_env = std::getenv("OSAI_EVAL_ONLY");
    const bool eval_only = assistant_only && eval_only_env != nullptr &&
            strcmp(eval_only_env, "1") == 0;
    if (eval_only) {
        if (weighted_alignment) {
            llama_opt_epoch_weighted(
                    ctx, dataset, result_train, result_eval, 0,
                    example_weights.data(), label_counts.data(), nullptr, nullptr);
        } else {
            llama_opt_epoch(ctx, dataset, result_train, result_eval, 0, nullptr, nullptr);
        }
        double eval_loss = 0.0;
        double eval_loss_unc = 0.0;
        ggml_opt_result_loss(result_eval, &eval_loss, &eval_loss_unc);
        LOG_INF("eval_loss=%.9g eval_loss_uncertainty=%.9g\n", eval_loss, eval_loss_unc);
        ggml_opt_result_free(result_train);
        ggml_opt_result_free(result_eval);
        ggml_opt_dataset_free(dataset);
        llama_backend_free();
        return std::isfinite(eval_loss) ? 0 : 1;
    }
    double best_train_loss = std::numeric_limits<double>::infinity();
    bool saved_best_adapter = false;

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        if (weighted_alignment) {
            llama_opt_epoch_weighted(
                    ctx, dataset, result_train, result_eval, idata_split,
                    example_weights.data(), label_counts.data(),
                    ggml_opt_epoch_callback_progress_bar,
                    ggml_opt_epoch_callback_progress_bar);
        } else {
            llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                            ggml_opt_epoch_callback_progress_bar,
                            ggml_opt_epoch_callback_progress_bar);
        }
        fprintf(stderr, "\n");

        double train_loss = 0.0;
        double train_loss_unc = 0.0;
        ggml_opt_result_loss(result_train, &train_loss, &train_loss_unc);
        LOG_INF("epoch=%u train_loss=%.9g train_loss_uncertainty=%.9g\n",
                lr.epoch + 1, train_loss, train_loss_unc);

        if (adapter_only && !weighted_alignment &&
                std::isfinite(train_loss) && train_loss < best_train_loss) {
            if (llama_adapter_lora_save_to_file(
                        llama_init->lora().front().get(), params.out_file.c_str()) != 0) {
                LOG_ERR("%s: failed to save best trained adapter\n", __func__);
                return 1;
            }
            best_train_loss = train_loss;
            saved_best_adapter = true;
            LOG_INF("checkpoint epoch=%u best_train_loss=%.9g\n", lr.epoch + 1, best_train_loss);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    if (adapter_only) {
        if (weighted_alignment) {
            if (llama_adapter_lora_save_to_file(
                        llama_init->lora().front().get(), params.out_file.c_str()) != 0) {
                LOG_ERR("%s: failed to save aligned adapter\n", __func__);
                return 1;
            }
        } else if (!saved_best_adapter && llama_adapter_lora_save_to_file(
                    llama_init->lora().front().get(), params.out_file.c_str()) != 0) {
            LOG_ERR("%s: failed to save trained adapter\n", __func__);
            return 1;
        }
    } else {
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return 0;
}
