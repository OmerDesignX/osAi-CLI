# Security

osAi accepts local custom-model and dataset paths. Selecting a missing official
tier permits one visible, checksum-verified download from the fixed
`OmerDesignX/osCode-Models` GitHub repository; `--no-download-model` or
`OSAI_OFFLINE=1` disables it. Training and inference force model hubs and
telemetry integrations offline, and bundled llama.cpp is built without CURL or
its server.

Do not use untrusted model or dataset files. Report security issues privately to
the project owner.
