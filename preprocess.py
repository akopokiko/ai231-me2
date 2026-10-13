#!/usr/bin/env python3
"""
Preprocesses and resamples audio files in the voice dataset to 16 kHz mono WAV format.
Bypasses torchcodec/ffmpeg issues by using soundfile and torchaudio/scipy resamplers.
"""

import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F_audio
from tqdm import tqdm

def resample_audio(audio_data, orig_sr, target_sr=16000):
    # Ensure audio is float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
        
    # Convert stereo to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    if orig_sr == target_sr:
        return audio_data
        
    tensor_audio = torch.from_numpy(audio_data).unsqueeze(0)
    resampled_tensor = F_audio.resample(tensor_audio, orig_sr, target_sr)
    return resampled_tensor.squeeze(0).numpy()

def main():
    parser = argparse.ArgumentParser(description="Preprocess and resample dataset audio to 16 kHz mono")
    parser.add_argument("--data_dir", type=str, default="data/voice_dataset", help="Directory containing dataset")
    parser.add_argument("--manifest", type=str, default="data/voice_dataset/manifest.csv", help="Input manifest CSV")
    parser.add_argument("--output_manifest", type=str, default="data/voice_dataset/manifest_16k.csv", help="Output 16kHz manifest CSV")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[-] Manifest not found: {args.manifest}")
        return

    df = pd.read_csv(args.manifest)
    print(f"[*] Resampling {len(df)} records to {args.target_sr} Hz...")

    new_records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resampling"):
        rel_path = str(row['path']).replace('\\', '/')
        audio_path = os.path.join(args.data_dir, rel_path)
        
        if not os.path.exists(audio_path):
            # Check relative to repo root
            if os.path.exists(rel_path):
                audio_path = rel_path
            else:
                print(f"[!] Warning: File missing: {audio_path}")
                continue

        try:
            audio_data, sr = sf.read(audio_path)
            resampled_audio = resample_audio(audio_data, sr, args.target_sr)
            
            # Destination path: <dir>_16k/<filename>
            parent_dir = os.path.dirname(audio_path)
            dir_name = os.path.basename(parent_dir)
            file_name = os.path.basename(audio_path)
            
            out_parent = os.path.join(os.path.dirname(parent_dir), f"{dir_name}_16k")
            os.makedirs(out_parent, exist_ok=True)
            
            out_path = os.path.join(out_parent, file_name)
            sf.write(out_path, resampled_audio, args.target_sr, subtype='PCM_16')
            
            new_row = row.to_dict()
            new_rel_path = os.path.relpath(out_path, start=args.data_dir).replace('\\', '/')
            new_row['path'] = new_rel_path
            new_row['duration_sec'] = len(resampled_audio) / float(args.target_sr)
            new_records.append(new_row)
        except Exception as e:
            print(f"[!] Error processing {audio_path}: {e}")

    new_df = pd.DataFrame(new_records)
    new_df.to_csv(args.output_manifest, index=False)
    print(f"[+] Resampling complete! Saved {len(new_df)} records to {args.output_manifest}")

if __name__ == "__main__":
    main()
