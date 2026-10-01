#How to execute?    goto powershell
# cd G:\AI_Tools\Kokoro
# .\venv\Scripts\Activate.ps1
# python merge_audio.py


from pathlib import Path
import soundfile as sf
import numpy as np


input_folder = Path(input("Enter input folder path: "))
output_folder = Path(input("Enter output folder path: "))

output_file = output_folder / f"{input_folder.name}.wav"

audio_files = sorted(input_folder.glob("*.wav"))

if not audio_files:
    print("No WAV files found")
    exit()

audio_data = []
for file in audio_files:
    print(f"Loading: {file.name}")
    audio, sample_rate = sf.read(file)

    if sample_rate != 24000:
        raise ValueError(f"Unexpected sample rate in {file.name}")

    audio_data.append(audio)

merged_audio = np.concatenate(audio_data)

sf.write(output_file, merged_audio, 24000)
print(f"Finished!... \nSaved to: {output_file}")
