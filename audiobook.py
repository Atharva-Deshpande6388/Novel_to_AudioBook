#How to execute?
# cd G:\AI_Tools\Kokoro
# .\venv\Scripts\Activate.ps1
# python audiobook.py

from pathlib import Path
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

INPUT_DIR = Path(input("Enter input directory: "))
OUTPUT_DIR = Path(input("Enter output directory: "))
VOICE = "af_river"
LANGUAGE = 'a'
MAX_CHARS = 1200
SAMPLE_RATE = 24000

def extract_text(path):
    if path.suffix.lower() == '.txt':
        return [(path.stem,path.read_text(encoding='utf-8-sig'))]
    if path.suffix.lower() == '.epub':
        book = epub.read_epub(str(path))
        chapters = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(),'html.parser')
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

def main():
    INPUT_DIR.mkdir(parents=True,exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in ('.txt','.epub')]
    if not files:
        print(f'Put a TXT or EPUB ebook in {INPUT_DIR}')
        return
    for i, path in enumerate(files,1):
        print(f"{i}. {path.name}")
        choice = int(input('Select Ebook Number: '))
        book_path = files[choice - 1]

        book_output = OUTPUT_DIR / book_path.stem
        book_output.mkdir(parents=True, exist_ok=True)
        chapters = extract_text(book_path)
        pipeline = KPipeline(lang_code = LANGUAGE)

        for chapter_num, (title, text) in enumerate(chapters, 1):
            safe_title = re.sub(r'[^W.-]+','_', title).strip('_')[:60]
            chunks = split_text(text)
            print(f"Chapter {chapter_num}: {title} ({len(chunks)} chunks)")

            for chunk_num, chunk in enumerate(chunks, 1):
                output = book_output / (
                    f"{chapter_num:03d}_{safe_title}_"
                    f"{chunk_num:04d}.wav"
                )
                if output.exists():
                    print(f'Skipping existing: {output.name}')
                    continue

                for _, _, audio in pipeline(chunk, voice=VOICE):
                    if hasattr(audio, 'detach'):
                        audio = audio.detach().cpu().numpy()
                    sf.write(str(output), np.asarray(audio), SAMPLE_RATE)
                print(f'Saved: {output.name}')
        print(f"Finished...\nAudio Saved in: {book_output}")

if __name__ == '__main__':
    main()
