#!/usr/bin/env python3
"""
Generates realistic hybrid UNKNOWN dataset samples:
- MUSAN speech slices (out-of-domain conversational speech)
- MUSAN ambient/background noise slices
- Synthetic ambient noise and near-silence
Assigns intent='UNKNOWN', slot='none', slot_value='none'.
"""

import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 16000

def get_audio_slice(file_path, target_samples):
    data, sr = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        import scipy.signal
        num_target = int(len(data) * SAMPLE_RATE / sr)
        data = scipy.signal.resample(data, num_target)

    if len(data) >= target_samples:
        start = random.randint(0, len(data) - target_samples)
        return data[start:start + target_samples]
    else:
        pad_len = target_samples - len(data)
        return np.pad(data, (0, pad_len), mode='constant')

def generate_unknown_samples(num_samples=300, data_dir="data/voice_dataset", musan_dir="data/musan"):
    out_dir = os.path.join(data_dir, "UNKNOWN_16k")
    os.makedirs(out_dir, exist_ok=True)

    # Collect MUSAN speech and noise files if available
    musan_speech = glob.glob(os.path.join(musan_dir, "speech", "**", "*.wav"), recursive=True)
    musan_noise = glob.glob(os.path.join(musan_dir, "noise", "**", "*.wav"), recursive=True)

    print(f"[*] Found {len(musan_speech)} MUSAN speech files, {len(musan_noise)} MUSAN noise files.")

    records = []
    for i in range(num_samples):
        # Duration between 1.5 and 2.5 seconds
        duration = random.uniform(1.8, 2.5)
        num_frames = int(duration * SAMPLE_RATE)
        
        # Determine source type
        source_type = random.choice(["musan_speech", "musan_noise", "synthetic_noise", "silence"])
        audio = None

        if source_type == "musan_speech" and musan_speech:
            chosen_file = random.choice(musan_speech)
            try:
                audio = get_audio_slice(chosen_file, num_frames)
            except Exception:
                pass

        elif source_type == "musan_noise" and musan_noise:
            chosen_file = random.choice(musan_noise)
            try:
                audio = get_audio_slice(chosen_file, num_frames)
            except Exception:
                pass

        if audio is None:
            if source_type == "silence":
                audio = np.random.randn(num_frames) * random.uniform(1e-5, 5e-4)
            else:
                # Synthetic ambient noise
                audio = np.random.randn(num_frames) * random.uniform(0.001, 0.02)

        # Normalize audio volume within reasonable bounds
        max_val = np.max(np.abs(audio))
        if max_val > 0.001:
            target_peak = random.uniform(0.1, 0.6)
            audio = (audio / max_val) * target_peak

        filename = f"unknown_{i:04d}.wav"
        filepath = os.path.join(out_dir, filename)
        sf.write(filepath, audio.astype(np.float32), SAMPLE_RATE, subtype='PCM_16')

        # Stratified train / val / test split (80 / 10 / 10)
        r = random.random()
        if r < 0.8:
            split = "train"
        elif r < 0.9:
            split = "val"
        else:
            split = "test"

        rel_path = os.path.relpath(filepath, start=data_dir).replace('\\', '/')
        records.append({
            "path": rel_path,
            "label": "UNKNOWN",
            "intent": "UNKNOWN",
            "speaker": "unknown_gen",
            "split": split,
            "phrase_id": f"v{i}",
            "variant_id": source_type,
            "transcript": "[UNKNOWN]",
            "slot": "none",
            "slot_value": "none",
            "duration_sec": round(duration, 3)
        })

    print(f"[+] Successfully generated {len(records)} UNKNOWN samples in {out_dir}")
    return records

def main():
    parser = argparse.ArgumentParser(description="Generate hybrid UNKNOWN dataset samples")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument("--data_dir", type=str, default="data/voice_dataset", help="Voice dataset directory")
    parser.add_argument("--musan_dir", type=str, default="data/musan", help="MUSAN dataset directory")
    args = parser.parse_args()

    generate_unknown_samples(args.num_samples, args.data_dir, args.musan_dir)

if __name__ == "__main__":
    main()
