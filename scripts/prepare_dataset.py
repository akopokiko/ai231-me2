#!/usr/bin/env python3
"""
Coordinates end-to-end dataset preparation and multi-speaker ingestion:
1. Resamples baseline dataset to 16kHz (if not already done).
2. Generates hybrid UNKNOWN samples (if not already done).
3. Ingests any custom voices from data/custom_voices/ (with 80/10/10 train/val/test split).
4. Generates unified manifest (data/manifest_unified.csv).
5. Exports synchronized intents.json (20 intents) and slots.json (19 slot values)
   into both data/ and deployment/.
"""

import os
import sys
import glob
import json
import argparse
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generate_unknowns import generate_unknown_samples

CANONICAL_SLOTS = {
    "TIMER": {
        "slot": "duration",
        "values": ["10 seconds", "30 seconds", "1 minute"]
    },
    "ALARM": {
        "slot": "time",
        "values": ["6 AM", "8 AM", "9 PM"]
    },
    "TEMPERATURE": {
        "slot": "degrees",
        "values": ["18 degrees", "22 degrees", "26 degrees"]
    },
    "BRIGHTNESS": {
        "slot": "percent",
        "values": ["20 percent", "60 percent", "100 percent"]
    },
    "COLOR": {
        "slot": "color",
        "values": ["red", "blue", "green"]
    },
    "CREATE_REMINDER": {
        "slot": "task",
        "values": ["drink water", "study", "exercise"]
    }
}

CANONICAL_INTENTS = [
    "ALARM",
    "BRIGHTNESS",
    "CALL",
    "COLOR",
    "CREATE_REMINDER",
    "LIGHT_OFF",
    "LIGHT_ON",
    "LIST_REMINDERS",
    "MESSAGE",
    "NEXT",
    "PAUSE",
    "PLAY_MUSIC",
    "STOP",
    "TEMPERATURE",
    "TIME",
    "TIMER",
    "UNKNOWN",
    "VOLUME_DOWN",
    "VOLUME_UP",
    "WEATHER"
]

def prepare_unified_dataset(base_data_dir="data/voice_dataset", custom_voices_dir="data/custom_voices"):
    print("=" * 60)
    print(" Preparing Unified Voice Control Dataset ")
    print("=" * 60)

    records = []

    # 1. Baseline dataset
    baseline_manifest_16k = os.path.join(base_data_dir, "manifest_16k.csv")
    baseline_manifest_raw = os.path.join(base_data_dir, "manifest.csv")

    if os.path.exists(baseline_manifest_16k):
        print(f"[*] Loading baseline 16kHz dataset from {baseline_manifest_16k}...")
        df_base = pd.read_csv(baseline_manifest_16k)
        for _, row in df_base.iterrows():
            d = row.to_dict()
            # Ensure path is relative to repo root
            p = str(d['path']).replace('\\', '/')
            if not p.startswith("data/"):
                p = os.path.join("data/voice_dataset", p).replace('\\', '/')
            d['path'] = p
            records.append(d)
        print(f"    Loaded {len(df_base)} baseline samples.")
    elif os.path.exists(baseline_manifest_raw):
        print("[!] Baseline raw manifest found, but 16kHz resampled manifest is missing.")
        print("    Running preprocess.py to resample baseline audio...")
        import subprocess
        subprocess.run(["python3", "preprocess.py"], check=True)
        return prepare_unified_dataset(base_data_dir, custom_voices_dir)
    else:
        print(f"[!] Warning: Baseline dataset not found at {base_data_dir}. Run scripts/download_baseline_data.py first.")

    # 2. Check or generate UNKNOWN samples
    unknown_dir = os.path.join(base_data_dir, "UNKNOWN_16k")
    unknown_wavs = glob.glob(os.path.join(unknown_dir, "*.wav")) if os.path.exists(unknown_dir) else []
    
    if len(unknown_wavs) < 100:
        print("[*] Generating hybrid UNKNOWN samples...")
        unknown_records = generate_unknown_samples(num_samples=300, data_dir=base_data_dir)
        for d in unknown_records:
            p = str(d['path']).replace('\\', '/')
            if not p.startswith("data/"):
                p = os.path.join("data/voice_dataset", p).replace('\\', '/')
            d['path'] = p
            records.append(d)
    else:
        print(f"[*] Found {len(unknown_wavs)} existing UNKNOWN samples.")
        for idx, u_path in enumerate(unknown_wavs):
            rel_p = os.path.relpath(u_path, start=".").replace('\\', '/')
            r = (idx % 10)
            split = "train" if r < 8 else ("val" if r == 8 else "test")
            try:
                info = sf.info(u_path)
                dur = round(info.duration, 3)
            except Exception:
                dur = 2.5
            records.append({
                "path": rel_p,
                "label": "UNKNOWN",
                "intent": "UNKNOWN",
                "speaker": "unknown_gen",
                "split": split,
                "phrase_id": f"v{idx}",
                "variant_id": "hybrid",
                "transcript": "[UNKNOWN]",
                "slot": "none",
                "slot_value": "none",
                "duration_sec": dur
            })

    # 3. Scan and ingest custom speaker recordings
    if os.path.exists(custom_voices_dir):
        speaker_dirs = [d for d in os.listdir(custom_voices_dir) if os.path.isdir(os.path.join(custom_voices_dir, d))]
        for spk in speaker_dirs:
            spk_dir = os.path.join(custom_voices_dir, spk)
            spk_manifest = os.path.join(spk_dir, "manifest.csv")
            if not os.path.exists(spk_manifest):
                spk_manifest = os.path.join(spk_dir, "manifest_16k.csv")
            if os.path.exists(spk_manifest):
                print(f"[*] Ingesting custom speaker '{spk}' manifest: {spk_manifest}...")
                df_spk = pd.read_csv(spk_manifest)
                # Assign 80/10/10 stratified split
                for idx, row in df_spk.iterrows():
                    d = row.to_dict()
                    r = (idx % 10)
                    d['split'] = "train" if r < 8 else ("val" if r == 8 else "test")
                    p = str(d['path']).replace('\\', '/')
                    if not p.startswith("data/"):
                        p = os.path.join("data/custom_voices", spk, p).replace('\\', '/')
                    d['path'] = p
                    records.append(d)
                print(f"    Added {len(df_spk)} samples from speaker '{spk}'.")

    # 3b. Also scan finetune_16k directly if present
    for alt_ft in ["finetune_16k", "data/finetune_16k"]:
        if os.path.exists(alt_ft):
            ft_manifest = os.path.join(alt_ft, "manifest_16k.csv")
            if not os.path.exists(ft_manifest):
                ft_manifest = os.path.join(alt_ft, "manifest.csv")
            if os.path.exists(ft_manifest):
                print(f"[*] Ingesting custom voice recordings from '{alt_ft}'...")
                df_ft = pd.read_csv(ft_manifest)
                for idx, row in df_ft.iterrows():
                    d = row.to_dict()
                    r = (idx % 10)
                    d['split'] = "train" if r < 8 else ("val" if r == 8 else "test")
                    p = str(d['path']).replace('\\', '/')
                    if not p.startswith(alt_ft):
                        p = os.path.join(alt_ft, p).replace('\\', '/')
                    d['path'] = p
                    records.append(d)
                print(f"    Added {len(df_ft)} samples from '{alt_ft}'.")

    if not records:
        print("[-] No records gathered. Please ensure dataset downloads or custom voice recordings exist.")
        return

    df_unified = pd.DataFrame(records)
    # Deduplicate by path
    df_unified.drop_duplicates(subset=['path'], keep='last', inplace=True)

    # Save unified manifest
    unified_manifest_path = "data/manifest_unified.csv"
    df_unified.to_csv(unified_manifest_path, index=False)
    print(f"\n[+] Unified manifest written to {unified_manifest_path} ({len(df_unified)} samples).")

    # Export canonical label and slot mappings
    for dest_dir in ["data", "deployment"]:
        os.makedirs(dest_dir, exist_ok=True)
        intents_file = os.path.join(dest_dir, "intents.json")
        slots_file = os.path.join(dest_dir, "slots.json")

        with open(intents_file, 'w', encoding='utf-8') as f:
            json.dump(CANONICAL_INTENTS, f, indent=2)

        with open(slots_file, 'w', encoding='utf-8') as f:
            json.dump(CANONICAL_SLOTS, f, indent=2)

    print("[+] Exported canonical 'intents.json' (20 intents) and 'slots.json' (19 slots) to data/ and deployment/.")

    # Print summary statistics
    print("\n--- Dataset Summary Statistics ---")
    print(f"Total Audio Samples : {len(df_unified)}")
    split_counts = df_unified['split'].value_counts().to_dict()
    print(f"Splits              : Train={split_counts.get('train', 0)}, Val={split_counts.get('val', 0)}, Test={split_counts.get('test', 0)}")
    print(f"Unique Speakers     : {df_unified['speaker'].nunique()} ({', '.join(df_unified['speaker'].unique()[:6])}...)")
    print(f"Unique Intents      : {df_unified['intent'].nunique()}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Prepare unified dataset with multi-speaker support")
    parser.add_argument("--base_data_dir", type=str, default="data/voice_dataset")
    parser.add_argument("--custom_voices_dir", type=str, default="data/custom_voices")
    args = parser.parse_args()

    prepare_unified_dataset(args.base_data_dir, args.custom_voices_dir)

if __name__ == "__main__":
    main()

