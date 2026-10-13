#!/usr/bin/env python3
"""
Downloads the baseline dataset from https://github.com/markandrian30/AI231/tree/main/MEX2/Data
using git sparse-checkout and places all files into data/voice_dataset/.
"""

import os
import shutil
import subprocess
import sys

def main():
    target_dir = os.path.abspath("data/voice_dataset")
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if already downloaded
    manifest_target = os.path.join(target_dir, "manifest.csv")
    if os.path.exists(manifest_target):
        print(f"[*] Baseline dataset already exists at {target_dir}")
        return

    print("[*] Downloading baseline dataset from GitHub repository...")
    repo_url = "https://github.com/markandrian30/AI231.git"
    temp_dir = os.path.abspath("temp_baseline_repo")

    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        print("[*] Cloning repository with sparse-checkout...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, temp_dir],
            check=True
        )

        subprocess.run(
            ["git", "sparse-checkout", "set", "MEX2/Data"],
            cwd=temp_dir,
            check=True
        )

        src_data = os.path.join(temp_dir, "MEX2", "Data")
        if not os.path.exists(src_data):
            raise RuntimeError(f"MEX2/Data not found in cloned repository: {src_data}")

        print(f"[*] Moving data files to {target_dir}...")
        for item in os.listdir(src_data):
            src_item = os.path.join(src_data, item)
            dst_item = os.path.join(target_dir, item)
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            shutil.move(src_item, dst_item)

        print(f"[+] Baseline dataset successfully placed in {target_dir}")

    except Exception as e:
        print(f"[-] Error downloading baseline dataset: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

