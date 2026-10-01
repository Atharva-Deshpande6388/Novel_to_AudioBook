#!/usr/bin/env python3
"""Merge the WAV chunks made by gpu_audiobook.py into one file per folder.

Point it at a folder that contains one subfolder per book/part and it writes
one merged WAV per subfolder, named after that subfolder. If the folder you
give it contains WAV files directly, those are merged into a single file too.

Examples:
    python merge_audio.py
    python merge_audio.py -i "audio" -o "merged"
    python merge_audio.py -i "audio\\Part_002" -o "merged" --pause 0.3
"""

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf

# Standard WAV tops out at 4 GiB; larger results are written as RF64
WAV_LIMIT = 4 * 1024 ** 3 - 1024 ** 2


def natural_key(path):
    """Sort Part_2 before Part_10, and chunk files in numeric order."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', path.name)]


def find_jobs(root, exclude):
    """Folders to merge: the root itself (if it holds WAVs) plus each subfolder."""
    jobs = []
    if any(root.glob('*.wav')):
        jobs.append(root)
    subfolders = sorted((p for p in root.iterdir() if p.is_dir()), key=natural_key)
    for sub in subfolders:
        if exclude is not None and sub.resolve() == exclude:
            continue
        if any(sub.glob('*.wav')):
            jobs.append(sub)
    return jobs


def merge_folder(folder, out_file, pause_seconds):
    """Stream every chunk in `folder` into `out_file`. Returns duration in seconds."""
    files = sorted(folder.glob('*.wav'), key=natural_key)

    # Validate everything first (reads headers only, so it's fast)
    first = sf.info(str(files[0]))
    rate, channels = first.samplerate, first.channels
    total_frames = 0
    for f in files:
        info = sf.info(str(f))
        if info.samplerate != rate or info.channels != channels:
            raise ValueError(
                f'{f.name} is {info.samplerate} Hz / {info.channels} ch, '
                f'expected {rate} Hz / {channels} ch')
        total_frames += info.frames

    pause = np.zeros((int(pause_seconds * rate), channels), dtype='int16')
    total_frames += len(pause) * (len(files) - 1)

    fmt = 'RF64' if total_frames * channels * 2 > WAV_LIMIT else 'WAV'
    if fmt == 'RF64':
        print('  Result exceeds 4 GB, using RF64 (still saved as .wav)')

    # Write to a temp file and rename, so a stopped run never leaves a
    # truncated file that looks finished
    tmp = out_file.with_suffix('.part')
    try:
        with sf.SoundFile(str(tmp), 'w', samplerate=rate, channels=channels,
                          subtype='PCM_16', format=fmt) as out:
            for i, f in enumerate(files, 1):
                data, _ = sf.read(str(f), dtype='int16')  # int16 -> PCM_16 is lossless
                out.write(data)
                if len(pause) and i < len(files):
                    out.write(pause)
                print(f'\r  [{i}/{len(files)}] {f.name[:55]:<55}', end='', flush=True)
        print()
        tmp.replace(out_file)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return total_frames / rate


def parse_args():
    p = argparse.ArgumentParser(
        description='Merge WAV chunks into one audio file per folder.')
    p.add_argument('-i', '--input', type=Path,
                   help='folder containing chunk folders (or WAV files directly); '
                        'asked interactively if omitted')
    p.add_argument('-o', '--output', type=Path,
                   help='folder for the merged files; asked interactively if omitted')
    p.add_argument('--pause', type=float, default=0.0,
                   help='seconds of silence between chunks (e.g. 0.3)')
    p.add_argument('--overwrite', action='store_true',
                   help='replace merged files that already exist '
                        '(by default they are skipped)')
    return p.parse_args()


def ask_path(label, default):
    raw = input(f'{label} [{default}]: ').strip().strip('"\'')
    return Path(raw) if raw else Path(default)


def fmt_duration(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m {s:02d}s'


def main():
    args = parse_args()

    input_dir = (args.input or ask_path('Input folder', 'output')).resolve()
    if not input_dir.is_dir():
        print(f'Input folder not found: {input_dir}')
        return

    default_out = input_dir.with_name(input_dir.name + '_merged')
    output_dir = (args.output or ask_path('Output folder', default_out)).resolve()

    jobs = find_jobs(input_dir, exclude=output_dir)
    if not jobs:
        print(f'No WAV files found in {input_dir} or its subfolders.')
        return
    if output_dir in [j.resolve() for j in jobs]:
        print('The output folder must be different from the folders being merged.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Found {len(jobs)} folder(s) to merge. Output: {output_dir}\n')

    merged, skipped, failed = 0, 0, 0
    total_seconds = 0.0
    for n, folder in enumerate(jobs, 1):
        out_file = output_dir / f'{folder.name}.wav'
        print(f'({n}/{len(jobs)}) {folder.name}')
        if out_file.exists() and not args.overwrite:
            print('  Already merged - skipping (use --overwrite to redo)')
            skipped += 1
            continue
        try:
            seconds = merge_folder(folder, out_file, args.pause)
        except ValueError as e:
            print(f'  Skipped: {e}')
            failed += 1
            continue
        total_seconds += seconds
        merged += 1
        print(f'  Saved: {out_file.name} ({fmt_duration(seconds)})')

    print(f'\nFinished. Merged {merged}, skipped {skipped}, failed {failed}'
          + (f', {fmt_duration(total_seconds)} of audio.' if merged else '.'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped by user. Completed merges are kept; rerun to continue.')
