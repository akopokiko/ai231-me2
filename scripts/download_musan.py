#!/usr/bin/env python3
"""
Downloads and extracts the MUSAN dataset (music, speech, and noise) from OpenSLR
into data/musan/.
"""

import os
import subprocess
import sys
import tarfile
import urllib.request
import shutil

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

def main():
    target_dir = os.path.abspath("data/musan")
    os.makedirs(target_dir, exist_ok=True)

    # Check if already extracted
    noise_dir = os.path.join(target_dir, "noise")
    speech_dir = os.path.join(target_dir, "speech")
    music_dir = os.path.join(target_dir, "music")
    if os.path.isdir(noise_dir) and os.path.isdir(speech_dir):
        print(f"[*] MUSAN dataset already exists at {target_dir}")
        return

    archive_path = os.path.join("data", "musan.tar.gz")

    if not os.path.exists(archive_path):
        print(f"[*] Downloading MUSAN from {MUSAN_URL}...")
        # Try using curl or wget if available for speed and resumability, else urllib
        cmd = None
        if shutil.which("curl"):
            cmd = ["curl", "-L", "-C", "-", "-o", archive_path, MUSAN_URL]
        elif shutil.which("wget"):
            cmd = ["wget", "-c", "-O", archive_path, MUSAN_URL]

        if cmd:
            subprocess.run(cmd, check=True)
        else:
            urllib.request.urlretrieve(MUSAN_URL, archive_path)
        print(f"[+] Downloaded MUSAN to {archive_path}")

    print(f"[*] Extracting {archive_path} to data/...")
    # The tar contains a top-level 'musan/' folder
    subprocess.run(["tar", "-xzf", archive_path, "-C", "data"], check=True)
    print(f"[+] MUSAN successfully extracted to {target_dir}")

if __name__ == "__main__":
    main()

