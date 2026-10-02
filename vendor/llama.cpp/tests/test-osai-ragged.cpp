#include "ggml-opt.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstdio>

int main() {
    constexpr int64_t width = 8;
    constexpr int64_t count = 2;
    const std::array<int32_t, 3> short_tokens = {11, 12, 13};
    const std::array<int32_t, 3> short_labels = {-1, 12, 13};
    const std::array<int32_t, 6> long_tokens = {21, 22, 23, 24, 25, 26};
    const std::array<int32_t, 6> long_labels = {-1, -1, -1, 24, 25, 26};
    const int32_t * data[count] = {short_tokens.data(), long_tokens.data()};
    const int32_t * labels[count] = {short_labels.data(), long_labels.data()};
    const int64_t lengths[count] = {short_tokens.size(), long_tokens.size()};
    auto * ragged = ggml_opt_dataset_init_ragged_i32(width, count, data, labels, lengths, 0);
    auto * padded = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, width, width, count, 1);
    auto * padded_data = static_cast<int32_t *>(ggml_opt_dataset_data(padded)->data);
    auto * padded_labels = static_cast<int32_t *>(ggml_opt_dataset_labels(padded)->data);
    for (int64_t row = 0; row < count; ++row) {
        for (int64_t column = 0; column < width; ++column) {
            padded_data[row*width + column] = 0;
            padded_labels[row*width + column] = -1;
        }
        for (int64_t column = 0; column < lengths[row]; ++column) {
            padded_data[row*width + column] = data[row][column];
            padded_labels[row*width + column] = labels[row][column];
        }
    }

    bool valid = ggml_opt_dataset_ndata(ragged) == count;
    valid = valid && ggml_opt_dataset_active_ubatches(ragged, 0, 2) == 2;
    valid = valid && ggml_opt_dataset_active_ubatches(ragged, 1, 2) == 3;
    for (int64_t row = 0; row < count; ++row) {
        std::array<int32_t, width> compact_data;
        std::array<int32_t, width> compact_labels;
        std::array<int32_t, width> fixed_data;
        std::array<int32_t, width> fixed_labels;
        ggml_opt_dataset_get_batch_host(
                ragged, compact_data.data(), sizeof(compact_data), compact_labels.data(), row);
        ggml_opt_dataset_get_batch_host(
                padded, fixed_data.data(), sizeof(fixed_data), fixed_labels.data(), row);
        valid = valid && compact_data == fixed_data && compact_labels == fixed_labels;
    }
    std::array<int32_t, width*count> compact_data_batch;
    std::array<int32_t, width*count> compact_labels_batch;
    std::array<int32_t, width*count> fixed_data_batch;
    std::array<int32_t, width*count> fixed_labels_batch;
    ggml_opt_dataset_get_batch_host(
            ragged, compact_data_batch.data(), sizeof(compact_data_batch), compact_labels_batch.data(), 0);
    ggml_opt_dataset_get_batch_host(
            padded, fixed_data_batch.data(), sizeof(fixed_data_batch), fixed_labels_batch.data(), 0);
    valid = valid && compact_data_batch == fixed_data_batch && compact_labels_batch == fixed_labels_batch;
    ggml_opt_dataset_free(ragged);
    ggml_opt_dataset_free(padded);
    std::printf("adaptive GGUF record padding: %s\n", valid ? "OK" : "FAIL");
    return valid ? 0 : 1;
}
