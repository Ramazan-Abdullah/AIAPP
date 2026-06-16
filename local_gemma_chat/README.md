# Local Gemma 3 Chat (offline, no Ollama)

A self-contained desktop chat app. The Gemma 3 model runs **in-process** via
`llama-cpp-python` — no Ollama, no separate server, no daemon. After the first
download the app works 100% offline on CPU.

## How it works

```
Tkinter desktop window  ──▶  llama-cpp-python (in the same process)  ──▶  gemma-3 GGUF file
```

The model file is downloaded once from Hugging Face into your local HF cache
(`~/.cache/huggingface`), then loaded straight into memory.

## Run

```bat
run.bat
```

or manually:

```bat
.venv\Scripts\python.exe chat_app.py
```

## Setup from scratch (if the .venv is missing)

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> Built on **Python 3.11** because `llama-cpp-python` has no prebuilt wheel for
> 3.14 yet. The venv keeps this isolated from your system Python.

## Choosing a model size

Edit `MODEL_SIZE` at the top of `chat_app.py`:

| Size  | Download | CPU feel              |
|-------|----------|-----------------------|
| `1b`  | ~0.8 GB  | fast (default)        |
| `4b`  | ~3.3 GB  | smarter, still usable |
| `12b` | ~7–8 GB  | needs lots of RAM     |
| `27b` | ~16 GB   | really wants a GPU    |

## Later: GPU acceleration

This build is CPU-only. For an NVIDIA GPU, reinstall with the CUDA wheel index
and set `n_gpu_layers=-1` in `chat_app.py`.
