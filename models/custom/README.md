# Custom local models

Place each custom model in its own named folder. A model may provide either or
both supported formats:

```text
models/custom/
└── my-model/
    ├── mlx/
    │   ├── config.json
    │   ├── model.safetensors.index.json
    │   └── model-00001-of-00001.safetensors
    └── gguf/
        └── my-model-Q4_K_M.gguf
```

Split GGUF models belong together in the `gguf` folder; osAi selects the
`00001` shard and verifies that every sibling shard exists. MLX weights must be
materialized and quantized. Symbolic network identifiers and unresolved Git LFS
pointers are rejected—only local files are accepted.

List and resolve custom models without loading weights:

```sh
.venv/bin/osai models
.venv/bin/osai select --custom my-model --engine auto
```

Train from the custom folder:

```sh
.venv/bin/osai train \
  --custom my-model \
  --engine auto \
  --data /absolute/path/to/jsonl-directory \
  --output artifacts/my-model-run
```

The name is restricted to letters, digits, dots, underscores, and hyphens so a
CLI argument cannot escape this directory.
