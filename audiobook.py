#How to execute?
# cd G:\AI_Tools\Kokoro 
# .\venv\Scripts\Activate.ps1
# python gpu_audiobook.py

import os
os.environ['HF_HOME'] = r'G:\Kokoro\hf_cache'

from pathlib import Path
import re
import multiprocessing as mp
import numpy as np
import soundfile as sf
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# INPUT_DIR = Path(r"G:\Kokoro\input")
# OUTPUT_DIR = Path(r"G:\Kokoro\output")
INPUT_DIR = Path(r"G:\novel\My Slain Dragon Bride Parts")
OUTPUT_DIR = Path(r"G:\novel\My Slain Dragon Bride Audio")
VOICE = "af_river"
LANGUAGE = 'a'
MAX_CHARS = 1200
SAMPLE_RATE = 24000
REPO_ID = 'hexgrad/Kokoro-82M'

VRAM_BUDGET_GB = 8.0
MAX_WORKERS = 4
HEADROOM = 1.25
FALLBACK_OVERHEAD_GB = 0.6

DISABLE_COMPLEX = True

_pipeline = None
_baseline_free = None
_device_name = ''
_total_mem = 0

def extract_text(path):
    if path.suffix.lower() == '.txt':
        return[(path.stem, path.read_text(encoding='utf-8-sig'))]

    if path.suffix.lower() == '.epub':
        book = epub.read_epub(str(path))
        chapters = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for tag in soup(['script','style']):
                tag.decompose()
            text = '\n'.join(
                line.strip() for line in soup.get_text('\n').splitlines()
                if line.strip()
            )
            if text:
                chapters.append((Path(item.get_name()).stem, text))
        return chapters
    raise ValueError('Please use a TXT or EPUB file')

def split_text(text, limit=MAX_CHARS):
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
            if cut<1:
                cut = limit
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()

        if not paragraph:
            continue

        if current and len(current) + len(paragraph) + 1 > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current} {paragraph}".strip()

    if current:
        chunks.append(current)
    return chunks

#== workers

def init_worker():
    global _pipeline, _baseline_free, _device_name, _total_mem
    import torch
    from kokoro import KModel, KPipeline

    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        raise RuntimeError(
            'Intel XPU not available - refusing to run on CPU. INSTALL THE XPU'
            'build of PyTorch and a current ARC driver'
        )
    torch.set_num_threads(1)

    idx = max(range(torch.xpu.device_count()),
              key = lambda i: torch.xpu.get_device_properties(i).total_memory)
    torch.xpu.set_device(idx)
    device = f'xpu:{idx}'
    _device_name = torch.xpu.get_device_name(idx)
    _total_mem = torch.xpu.get_device_properties(idx).total_memory

    try:
        _baseline_free = torch.xpu.mem_get_info()[0]
    except Exception:
        _baseline_free = None

    kModel = KModel(repo_id = REPO_ID, disable_complex=DISABLE_COMPLEX).to(device).eval()
    _pipeline = KPipeline(lang_code = LANGUAGE, repo_id = REPO_ID, model = kModel)

def _render(chunk):
    parts = []
    for _, _, audio in _pipeline(chunk, voice=VOICE):
        if audio is None:
            continue
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().numpy()
        parts.append(np.asarray(audio).reshape(-1))
    return parts

def probe(chunk):
    """Synthesize one chunk and report this worker's total VRAM footprint"""
    import torch
    _render(chunk)
    torch.xpu.synchronize()
    used = None
    if _baseline_free is not None:
        try:
            used = _baseline_free - torch.xpu.mem_get_info()[0]
        except Exception:
            used = None
    if used is None or used <= 0:
        used = torch.xpu.max_memory_reserved() + int(FALLBACK_OVERHEAD_GB*1024**3)
    return {'used':int(used), 'device':_device_name, 'total':int(_total_mem)}

def synthesize(task):
    import torch
    out_path, chunk = task
    parts = _render(chunk)
    torch.xpu.empty_cache()

    if not parts:
        return out_path.name, False

    tmp = out_path.with_suffix('.part')
    sf.write(str(tmp), np.concatenate(parts), SAMPLE_RATE, format='WAV')
    tmp.replace(out_path)
    return out_path.name, True

#== main

def main():
    import torch
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print('No Intel XPU detected. Install the XPU build of PyTorch:\n'
              '  pip install torch torchaudio --index-url '
              'https://download.pytorch.org/whl/xpu\n'
              'and make sure your ARC driver is up to date')
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [p for p in INPUT_DIR.iterdir()
             if p.suffix.lower() in ('.txt','.epub')]
    if not files:
        print(f'Put a TXT or EPUB ebook in {INPUT_DIR}')
        return

    for i, path in enumerate(files, 1):
        print(f"{i}. {path.name}")
    choice = int(input('Select ebook Number: '))
    book_path = files[choice - 1]

    book_output = OUTPUT_DIR/book_path.stem
    book_output.mkdir(parents=True, exist_ok=True)
    chapters = extract_text(book_path)

    tasks = []
    for chapter_num, (title, text) in enumerate(chapters, 1):
        safe_title = re.sub(r'[^\w.-]+', '_', title).strip('_')[:60]
        chunks = split_text(text)
        print(f"Chapter {chapter_num}: {title} ({len(chunks)} chunks)")
        for chunk_num, chunk in enumerate(chunks, 1):
            out = book_output/f'{chapter_num:03d}_{safe_title}_{chunk_num:04d}.wav'
            if not out.exists():
                tasks.append((out, chunk))

    if not tasks:
        print("Nothing to do - all chunks already exists")
        return

    ctx = mp.get_context('spawn')

    longest = max(tasks, key=lambda t: len(t[1]))[1]
    print('Measuring per-worker VRAM footprint...')
    with ctx.Pool(1, initializer=init_worker) as pool:
        info = pool.apply(probe, (longest,))

    gib = 1024 ** 3
    per_worker = max(info['used'], int(0.5 * gib))
    budget = min(VRAM_BUDGET_GB * gib, info['total'] * 0.95)
    workers = int(max(1, min(MAX_WORKERS, budget // (per_worker * HEADROOM))))

    print(f"GPU: {info['device']} ({info['total'] / gib:.1f} GB)")
    print(f'Per-worker footprint: {per_worker / gib:.2f} GB ->'
          f'{workers} worker(s), about {workers * per_worker / gib:.1f} GB'
          f'of the {VRAM_BUDGET_GB:.0f} GB budget')

    print(f"Synthesizing {len(tasks)} chunks...")
    with ctx.Pool(workers, initializer=init_worker) as pool:
        for done, (name, ok) in enumerate(
            pool.imap_unordered(synthesize, tasks, chunksize=1),1):
            status = 'Saved' if ok else 'No audio for'
            print(f'[{done}/{len(tasks)}] {status}: {name}')

    print(f"finished. Audio saved in: {book_output}")


if __name__ == '__main__':
    main()
