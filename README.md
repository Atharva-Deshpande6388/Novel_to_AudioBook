# Kokoro GPU Audiobook Maker

Convert `.txt` and `.epub` ebooks into audio with [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M), running **entirely on the GPU** (Intel Arc via XPU, or NVIDIA via CUDA). Several worker processes share the GPU while total VRAM use stays under a budget you set, so the machine stays usable while a book converts.

Tested on an Intel Arc B580 on Windows. The NVIDIA CUDA path uses the same code but has not been tested.

## Features

- GPU only: refuses to run on the CPU
- Multiple workers on one GPU, with the count chosen automatically from a measured VRAM footprint
- Resumable: finished chunks are kept, and interrupted runs pick up where they stopped
- Safe to stop: press Ctrl+C once; no half-written audio files are left behind
- Chapters and chunks are numbered in the filenames, so sorting by name gives the correct order

## Requirements

- Python 3.10 to 3.12
- Intel Arc GPU with a current driver, **or** an NVIDIA GPU
- About 330 MB of disk for the model (downloaded on first run)

## Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Replace the CPU build of PyTorch that kokoro installs with one for your GPU.
# Intel Arc:
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/xpu
# NVIDIA: use the command from https://pytorch.org/get-started/locally/
```

Verify the GPU is visible (Intel Arc shown):

```powershell
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

You want a version ending in `+xpu` and `True`. If it ends in `+cpu`, the wrong build is installed.

> Installing or upgrading packages that depend on PyTorch (including `kokoro`) can silently replace the GPU build with the CPU build. If that happens, rerun the PyTorch install above.

## Usage

Run with no arguments to be prompted for the input and output folders:

```powershell
python gpu_audiobook.py
```

Or pass everything on the command line:

```powershell
python gpu_audiobook.py -i books -o audio --voice af_heart
python gpu_audiobook.py -i "books\My Book.epub" -o audio --max-workers 2
```

`--input` can be a folder (you pick a book from the list) or a single file. Audio is written to `<output>/<book name>/`.

## Options

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | asked | Folder with `.txt`/`.epub` files, or one file |
| `-o`, `--output` | asked | Folder for the generated audio |
| `--voice` | `af_heart` | Kokoro voice name |
| `--lang` | `a` | Kokoro language code (`a` = American English) |
| `--max-chars` | `1200` | Maximum characters per chunk |
| `--device` | `auto` | `auto`, `xpu` or `cuda` |
| `--vram-budget` | `8.0` | VRAM ceiling in GB for all workers combined |
| `--max-workers` | `4` | Maximum worker processes |
| `--hf-cache` | default | Where Hugging Face models are cached |
| `--offline` | off | Never contact Hugging Face (needs a populated cache) |
| `--enable-complex` | off | Use the complex-number STFT; may be unsupported on some GPUs |

## How the VRAM budget works

PyTorch's XPU backend has no hard per-process memory cap, so the script budgets instead. It starts one probe worker, synthesizes your longest chunk, measures the worker's total VRAM footprint, then picks `workers = budget / (footprint x 1.25)`, capped by `--max-workers`. Each worker also frees its allocator cache after every chunk. The budget covers this script only, not Windows or other applications.

## Merging chunks

`merge_audio.py` joins the chunk files into one WAV per book or part. Point it at the folder that contains the book folders (your gpu_audiobook output folder):

```powershell
python merge_audio.py                       # prompts for the folders
python merge_audio.py -i audio -o merged
python merge_audio.py -i audio -o merged --pause 0.3
```

Each subfolder of `audio` becomes `merged/<folder name>.wav`. Folders that are already merged are skipped unless you pass `--overwrite`, and `--pause` inserts a short silence between chunks. Files over 4 GB are written as RF64 automatically.

## Notes

- The first run downloads the model and the chosen voice, then everything loads from the local cache.
- `--max-workers` is the main speed/headroom trade-off: more workers are faster until the GPU or CPU saturates, fewer leave more room for other work.
- Only convert books you have the right to use. Do not commit ebooks or generated audio to this repository.
