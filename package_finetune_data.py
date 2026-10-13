#!/usr/bin/env python3
"""
Packages existing recordings from finetune_16k/ into custom_voice_s_user.zip.
Run this on your laptop to bundle your recorded voice files for transfer to DGX!
"""

import os
import glob
import zipfile
import pandas as pd

def main():
    source_dir = "finetune_16k"
    if not os.path.exists(source_dir):
        # Check if running from within another directory
        if os.path.exists("../finetune_16k"):
            source_dir = "../finetune_16k"
        else:
            print(f"[-] Directory '{source_dir}' not found in current folder ({os.getcwd()}).")
            return

    wav_files = glob.glob(os.path.join(source_dir, "**", "*.wav"), recursive=True)
    print(f"[*] Found {len(wav_files)} .wav audio file(s) in '{source_dir}'.")

    if not wav_files:
        print("[!] No .wav files found inside finetune_16k!")
        return

    speaker_id = input("Enter your Speaker ID [Default: s_user]: ").strip() or "s_user"
    zip_name = f"custom_voice_{speaker_id}.zip"

    manifest_file = os.path.join(source_dir, "manifest_16k.csv")
    if not os.path.exists(manifest_file):
        manifest_file = os.path.join(source_dir, "manifest.csv")

    print(f"[*] Packaging {len(wav_files)} audio files into '{zip_name}'...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Include manifest
        if os.path.exists(manifest_file):
            zf.write(manifest_file, arcname=os.path.join(speaker_id, "manifest.csv"))
            print("  [+] Included manifest.csv")

        # Include all wav files
        for wav_path in wav_files:
            rel = os.path.relpath(wav_path, start=source_dir)
            arcname = os.path.join(speaker_id, rel)
            zf.write(wav_path, arcname=arcname)

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"[+] SUCCESS! Created '{zip_name}' ({size_mb:.2f} MB)")
    print(f"    Contains {len(wav_files)} audio files for speaker '{speaker_id}'.")
    print(f"    Transfer this zip to DGX and unzip into 'data/custom_voices/'!")
    print("=" * 60)

if __name__ == "__main__":
    main()

