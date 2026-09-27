#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf -p \"Hello world\"\n", argv[0]);
    LOG("\n  %s -m your_model.gguf -f prompt.txt --ids\n", argv[0]);
    LOG("\n  cat prompt.txt | %s -m your_model.gguf --stdin --show-count\n", argv[0]);
    LOG("\n");
}

//
// Function: write_utf8_cstr_to_stdout(const char *) -> <writes to stdout>
//
// writes a string to standard output; taking into account that on Windows
// to display correctly you have to use special handling. Works even if the
// user has not set a unicode code page on a Windows cmd.exe.
//
// In case of invalid UTF-8, invalid_utf8 is set to true on Windows, and something
// a human-readable is written instead.
//
// On non-Windows systems, simply printfs() the string.
static void write_utf8_cstr_to_stdout(const char * str, bool & invalid_utf8) {
        invalid_utf8 = false;

#if defined(_WIN32)
        // Are we in a console?
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;

        // According to Microsoft docs:
        // "WriteConsole fails if it is used with a standard handle that is redirected to a file."
        // Also according to the docs, you can use GetConsoleMode to check for that.
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            printf("%s", str);
            return;
        }

        // MultiByteToWideChar reports an error if str is empty, don't report
        // them as invalid_utf8.
        if (*str == 0) {
            return;
        }
        int length_needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, strlen(str), NULL, 0);
        if (length_needed == 0) {
            DWORD err = GetLastError();
            if (err == ERROR_NO_UNICODE_TRANSLATION) {
                invalid_utf8 = true;
                int len = strlen(str);
                printf("<");
                for (int i = 0; i < len; ++i) {
                    if (i > 0) {
                        printf(" ");
                    }
                    printf("%02x", (uint8_t) str[i]);
                }
                printf(">");
                return;
            }
            GGML_ABORT("MultiByteToWideChar() failed in an unexpected way.");
        }

        LPWSTR wstr = (LPWSTR) calloc(length_needed+1, sizeof(*wstr));
        GGML_ASSERT(wstr);

        MultiByteToWideChar(CP_UTF8, 0, str, strlen(str), wstr, length_needed);
        WriteConsoleW(hConsole, wstr, length_needed, NULL, NULL);

        free(wstr);
#else
        // TODO: reporting invalid_utf8 would be useful on non-Windows too.
        // printf will silently just write bad unicode.
        printf("%s", str);
#endif
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    bool osai_record_counts = false;
    std::vector<char *> parsed_argv;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--osai-record-counts") == 0) {
            osai_record_counts = true;
        } else {
            parsed_argv.push_back(argv[i]);
        }
    }
    if (!common_params_parse((int) parsed_argv.size(), parsed_argv.data(),
                             params, LLAMA_EXAMPLE_TOKENIZE, print_usage)) {
        return 1;
    }

    // -f and -p both land in params.prompt; -f also sets prompt_file. -f and -p
    // resolve like the other tools (no mutual exclusion), --stdin takes precedence.
    const bool use_stdin = params.tokenize_stdin || osai_record_counts;
    const bool use_file  = !params.prompt_file.empty();

    // must have some prompt
    if (!use_stdin && !use_file && params.prompt.empty()) {
        LOG_ERR("error: must specify one of: --stdin, --file or --prompt\n");
        return 1;
    }

    std::string prompt;
    if (use_file) {
        // read the file verbatim: common's -f handler strips a single trailing
        // newline, but for a tokenizer the input bytes must be preserved exactly
        // (a trailing newline is itself a token). escapes are applied locally
        // to match the behavior of -p/--prompt and --stdin.
        std::ifstream in(params.prompt_file, std::ios::binary);
        if (!in) {
            LOG_ERR("error: could not open file '%s' for reading\n", params.prompt_file.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << in.rdbuf();
        prompt = ss.str();
        if (params.escape) {
            string_process_escapes(prompt);
        }
    } else if (!use_stdin) {
        // -p/--prompt is already escape-processed by common_params_parse()
        // (controlled by --escape/--no-escape), so use it verbatim here.
        prompt = params.prompt;
    }
    // else: we read stdin *after* loading the model (early exit if the
    // model cannot be loaded, which is a nicer user experience)

    llama_backend_init();

    // load only the vocabulary (no weights), since tokenizing does not need them
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: could not load model from file '%s'.\n", params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (osai_record_counts) {
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        size_t records = 0;
        size_t largest = 0;
        while (true) {
            unsigned char length_bytes[8];
            std::cin.read(reinterpret_cast<char *>(length_bytes), 8);
            if (std::cin.gcount() == 0 && std::cin.eof()) {
                break;
            }
            if (std::cin.gcount() != 8) {
                LOG_ERR("error: incomplete osAi record length\n");
                llama_model_free(model);
                return 1;
            }
            uint64_t length = 0;
            for (int i = 0; i < 8; ++i) {
                length |= (uint64_t) length_bytes[i] << (8 * i);
            }
            if (length > 128 * 1024 * 1024) {
                LOG_ERR("error: osAi record is too large to tokenize safely\n");
                llama_model_free(model);
                return 1;
            }
            std::string record((size_t) length, '\0');
            std::cin.read(record.data(), (std::streamsize) length);
            if ((uint64_t) std::cin.gcount() != length) {
                LOG_ERR("error: incomplete osAi record content\n");
                llama_model_free(model);
                return 1;
            }
            const auto tokens = common_tokenize(
                vocab, record, llama_vocab_get_add_bos(vocab), false);
            if (tokens.size() > largest) {
                largest = tokens.size();
            }
            ++records;
        }
        printf("osai_max_tokens=%zu records=%zu\n", largest, records);
        llama_model_free(model);
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: could not create context.\n");
        return 1;
    }

    // read entire prompt from stdin?
    if (params.tokenize_stdin) {
        std::stringstream stdin_buffer;
        stdin_buffer << std::cin.rdbuf();
        if (std::cin.fail()) {
            LOG_ERR("error: could not read the entire standard input.\n");
            return 1;
        }

        prompt = stdin_buffer.str();

        // stdin is not seen by common_params_parse(), so apply escape handling
        // here to match the behavior of -p/--prompt and -f/--file.
        if (params.escape) {
            string_process_escapes(prompt);
        }
    }

    const bool model_wants_add_bos = llama_vocab_get_add_bos(vocab);
    const bool add_bos      = model_wants_add_bos && !params.tokenize_no_bos;
    const bool parse_special = params.parse_special;

    std::vector<llama_token> tokens;
    tokens = common_tokenize(vocab, prompt, add_bos, parse_special);

    if (params.tokenize_ids) {
        printf("[");
    }

    for (int i = 0; i < (int) tokens.size(); i++) {
        if (params.tokenize_ids) {
            if (i > 0) {
                printf(", ");
            }
            printf("%d", tokens[i]);
        } else {
            bool invalid_utf8 = false;
            printf("%6d -> '", tokens[i]);
            write_utf8_cstr_to_stdout(common_token_to_piece(ctx, tokens[i]).c_str(), invalid_utf8);
            if (invalid_utf8) {
                printf("' (utf-8 decode failure)\n");
            } else {
                printf("'\n");
            }
        }
    }

    if (params.tokenize_ids) {
        printf("]\n");
    }

    if (params.tokenize_show_count) {
        printf("Total number of tokens: %zu\n", tokens.size());
    }

    // silence valgrind
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
