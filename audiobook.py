#!/usr/bin/env python3
"""Convert TXT/EPUB ebooks to audio chunks with Kokoro TTS, GPU only.

Runs several worker processes on one GPU (Intel Arc via XPU, or NVIDIA via
CUDA) while keeping total VRAM use under a budget. Finished chunks are kept,
so an interrupted run resumes where it stopped.

Examples:
    python gpu_audiobook.py
    python gpu_audiobook.py -i books -o audio --voice af_heart
    python gpu_audiobook.py -i "books\\My Book.epub" -o audio --max-workers 2
"""

import argparse
import multiprocessing as mp
import os
import re
import signal
from pathlib import Path

import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

SAMPLE_RATE = 24000
REPO_ID = 'hexgrad/Kokoro-82M'
HEADROOM = 1.25               # safety multiplier on measured per-worker VRAM
FALLBACK_OVERHEAD_GB = 0.6    # used only if mem_get_info() is unavailable

# Per-worker state (each worker process has its own copy)
_pipeline = None
_baseline_free = None
_device_name = ''
_total_mem = 0
_cfg = {}


#= text prep =#

def extract_text(path):
    suffix = path.suffix.lower()
    if suffix == '.txt':
        return [(path.stem, path.read_text(encoding='utf-8-sig'))]

    if suffix == '.epub':
        book = epub.read_epub(str(path))
        chapters = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = '\n'.join(
                line.strip() for line in soup.get_text('\n').splitlines()
                if line.strip()
            )
            if text:
                chapters.append((Path(item.get_name()).stem, text))
        return chapters

    raise ValueError('Please use a TXT or EPUB file.')


def split_text(text, limit):
    paragraphs = re.split(r'\n+', text)
    chunks = []
    current = ''

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > limit:
            if current:
                chunks.append(current)
                current = ''
            cut = paragraph.rfind(' ', 0, limit)
            if cut < 1:
                cut = limit
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()

        if not paragraph:
            continue

        if current and len(current) + len(paragraph) + 1 > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = f'{current} {paragraph}'.strip()

    if current:
        chunks.append(current)
    return chunks


#= workers =#

def get_backend(torch, requested):
    """Return (name, module) for the first available GPU backend, or (None, None)."""
    order = ['xpu', 'cuda'] if requested == 'auto' else [requested]
    for name in order:
        module = getattr(torch, name, None)
        if module is not None and module.is_available():
            return name, module
    return None, None


def init_worker(cfg):
    """Runs once per worker: load Kokoro onto the GPU (never the CPU)."""
    global _pipeline, _baseline_free, _device_name, _total_mem, _cfg
    # Workers ignore Ctrl+C; the main process stops them cleanly instead
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    from kokoro import KModel, KPipeline

    _cfg = cfg
    name, backend = get_backend(torch, cfg['device'])
    if backend is None:
        raise RuntimeError(f"GPU backend '{cfg['device']}' not available - "
                           'refusing to run on CPU.')

    torch.set_num_threads(1)  # avoid CPU oversubscription across workers

    # If several GPUs exist (e.g. iGPU + Arc), use the one with the most memory
    idx = max(range(backend.device_count()),
              key=lambda i: backend.get_device_properties(i).total_memory)
    backend.set_device(idx)
    device = f'{name}:{idx}'
    _device_name = backend.get_device_name(idx)
    _total_mem = backend.get_device_properties(idx).total_memory

    try:
        _baseline_free = backend.mem_get_info()[0]
    except Exception:
        _baseline_free = None

    kmodel = KModel(repo_id=cfg['repo_id'],
                    disable_complex=cfg['disable_complex']).to(device).eval()
    _pipeline = KPipeline(lang_code=cfg['lang'], repo_id=cfg['repo_id'],
                          model=kmodel)


def _render(chunk):
    parts = []
    for _, _, audio in _pipeline(chunk, voice=_cfg['voice']):
        if audio is None:
            continue
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().numpy()
        parts.append(np.asarray(audio).reshape(-1))
    return parts


def probe(chunk):
    """Synthesize one chunk and report this worker's total VRAM footprint."""
    import torch
    backend = getattr(torch, _cfg['device'])
    _render(chunk)
    backend.synchronize()
    used = None
    if _baseline_free is not None:
        try:
            used = _baseline_free - backend.mem_get_info()[0]
        except Exception:
            used = None
    if used is None or used <= 0:
        used = (backend.max_memory_reserved()
                + int(FALLBACK_OVERHEAD_GB * 1024 ** 3))
    return {'used': int(used), 'device': _device_name, 'total': int(_total_mem)}


def synthesize(task):
    import torch
    out_path, chunk = task
    parts = _render(chunk)
    getattr(torch, _cfg['device']).empty_cache()  # keep VRAM within budget

    if not parts:
        return out_path.name, False

    tmp = out_path.with_suffix('.part')
    sf.write(str(tmp), np.concatenate(parts), SAMPLE_RATE, format='WAV')
    tmp.replace(out_path)  # only complete files ever get the .wav name
    return out_path.name, True


#= CLI =#

def parse_args():
    p = argparse.ArgumentParser(
        description='Convert TXT/EPUB ebooks to audio chunks with Kokoro TTS '
                    '(GPU only).')
    p.add_argument('-i', '--input', type=Path,
                   help='folder containing .txt/.epub files, or a single file '
                        '(asked interactively if omitted)')
    p.add_argument('-o', '--output', type=Path,
                   help='folder for the generated audio '
                        '(asked interactively if omitted)')
    p.add_argument('--voice', default='af_heart', help='Kokoro voice name')
    p.add_argument('--lang', default='a',
                   help="Kokoro language code ('a' = American English)")
    p.add_argument('--max-chars', type=int, default=1200,
                   help='maximum characters per chunk')
    p.add_argument('--device', choices=['auto', 'xpu', 'cuda'], default='auto',
                   help='GPU backend: Intel XPU or NVIDIA CUDA')
    p.add_argument('--vram-budget', type=float, default=8.0,
                   help='VRAM ceiling in GB for all workers combined')
    p.add_argument('--max-workers', type=int, default=4,
                   help='maximum worker processes sharing the GPU')
    p.add_argument('--hf-cache', type=Path,
                   help='where Hugging Face models are cached '
                        '(default: HF_HOME or the standard cache)')
    p.add_argument('--offline', action='store_true',
                   help='never contact Hugging Face (needs a populated cache)')
    p.add_argument('--enable-complex', action='store_true',
                   help='use the complex-number STFT (may be faster, may be '
                        'unsupported on some GPUs)')
    return p.parse_args()


def ask_path(label, default):
    raw = input(f'{label} [{default}]: ').strip().strip('"\'')
    return Path(raw) if raw else Path(default)


def choose_book(files):
    if len(files) == 1:
        print(f'Using: {files[0].name}')
        return files[0]
    for i, path in enumerate(files, 1):
        print(f'{i}. {path.name}')
    while True:
        raw = input('Select ebook number: ').strip()
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        print(f'Please enter a number from 1 to {len(files)}.')


#= main =#

def main():
    args = parse_args()

    # Must be set before Hugging Face is imported; spawned workers inherit it
    if args.hf_cache:
        os.environ['HF_HOME'] = str(args.hf_cache)
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'

    import torch
    backend_name, _ = get_backend(torch, args.device)
    if backend_name is None:
        print('No supported GPU detected (Intel XPU or NVIDIA CUDA).\n'
              'For Intel Arc install the XPU build of PyTorch:\n'
              '  pip install torch torchaudio --index-url '
              'https://download.pytorch.org/whl/xpu\n'
              'and make sure your GPU driver is up to date.')
        return

    input_path = args.input or ask_path('Input folder or ebook file', 'input')
    output_dir = args.output or ask_path('Output folder', 'output')

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(p for p in input_path.iterdir()
                       if p.suffix.lower() in ('.txt', '.epub'))
    else:
        print(f'Input path not found: {input_path}')
        return
    if not files:
        print(f'No .txt or .epub files found in {input_path}')
        return

    book_path = choose_book(files)
    book_output = output_dir / book_path.stem
    book_output.mkdir(parents=True, exist_ok=True)

    cfg = {
        'voice': args.voice,
        'lang': args.lang,
        'repo_id': REPO_ID,
        'disable_complex': not args.enable_complex,
        'device': backend_name,
    }

    chapters = extract_text(book_path)

    # Queue every chunk of every chapter first, skipping finished files
    tasks = []
    for chapter_num, (title, text) in enumerate(chapters, 1):
        safe_title = re.sub(r'[^\w.-]+', '_', title).strip('_')[:60]
        chunks = split_text(text, args.max_chars)
        print(f'Chapter {chapter_num}: {title} ({len(chunks)} chunks)')
        for chunk_num, chunk in enumerate(chunks, 1):
            out = book_output / f'{chapter_num:03d}_{safe_title}_{chunk_num:04d}.wav'
            if not out.exists():
                tasks.append((out, chunk))

    if not tasks:
        print('Nothing to do - all chunks already exist.')
        return

    ctx = mp.get_context('spawn')

    # Step 1: measure one worker's real VRAM footprint on the longest chunk
    longest = max(tasks, key=lambda t: len(t[1]))[1]
    print('Measuring per-worker VRAM footprint...')
    with ctx.Pool(1, initializer=init_worker, initargs=(cfg,)) as pool:
        info = pool.apply(probe, (longest,))

    gib = 1024 ** 3
    per_worker = max(info['used'], int(0.5 * gib))
    budget = min(args.vram_budget * gib, info['total'] * 0.95)
    workers = int(max(1, min(args.max_workers, budget // (per_worker * HEADROOM))))

    print(f"GPU: {info['device']} ({info['total'] / gib:.1f} GB)")
    print(f'Per-worker footprint: {per_worker / gib:.2f} GB -> '
          f'{workers} worker(s), about {workers * per_worker / gib:.1f} GB '
          f'of the {args.vram_budget:.0f} GB budget')

    # Step 2: synthesize
    print(f'Synthesizing {len(tasks)} chunks... (press Ctrl+C once to stop safely)')
    pool = ctx.Pool(workers, initializer=init_worker, initargs=(cfg,))
    try:
        for done, (name, ok) in enumerate(
                pool.imap_unordered(synthesize, tasks, chunksize=1), 1):
            status = 'Saved' if ok else 'No audio for'
            print(f'[{done}/{len(tasks)}] {status}: {name}')
        pool.close()
    except KeyboardInterrupt:
        print('\nInterrupted - stopping workers. Finished chunks are kept; '
              'rerun the same command to resume.')
        pool.terminate()
        pool.join()
        return
    pool.join()

    print(f'Finished. Audio saved in: {book_output}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped by user.')
