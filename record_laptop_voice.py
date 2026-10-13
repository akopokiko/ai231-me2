#!/usr/bin/env python3
"""
Portable Voice Recording Utility for Laptop (Windows / Mac / Linux)
Records custom voice samples at 16 kHz mono for 19 intents + slotted variations.
Supports multiple speakers (e.g., s_user, s_friend1, s_friend2) and packages recordings into a zip.
"""

import os
import sys
import time
import wave
import zipfile

try:
    import sounddevice as sd
except ImportError:
    print("[!] sounddevice is required. Install via: pip install sounddevice numpy")
    sys.exit(1)

import numpy as np

SAMPLE_RATE = 16000
DURATION = 2.5
CHANNELS = 1

FIXED_PHRASES = {
    "PLAY_MUSIC": ["Play music", "Start music", "Play some music"],
    "VOLUME_UP": ["Volume up", "Increase the volume", "Turn the volume up"],
    "VOLUME_DOWN": ["Volume down", "Lower the volume", "Turn the volume down"],
    "NEXT": ["Next song", "Skip song", "Play next song"],
    "PAUSE": ["Pause", "Pause audio", "Pause song"],
    "STOP": ["Stop", "Stop playing", "End playback"],
    "LIGHT_ON": ["Lights on", "Power on the lights", "Turn on the lights"],
    "LIGHT_OFF": ["Lights out", "Kill the lights", "Shut off the lights"],
    "WEATHER": ["Weather", "What's the weather?", "Tell me the weather"],
    "TIME": ["Time", "What time is it?", "Tell me the time"],
    "CALL": ["Call", "Make a call", "Make a phone call"],
    "MESSAGE": ["Message", "Send a message", "Send my message"],
    "LIST_REMINDERS": ["Reminders", "Show my reminders", "List my reminders"],
}

SLOTTED_PHRASES = {
    "BRIGHTNESS": {
        "phrases": ["Brightness {percent}", "Adjust brightness to {percent}", "Brightness level {percent}"],
        "values": ["20 percent", "60 percent", "100 percent"],
        "slot_name": "percent"
    },
    "COLOR": {
        "phrases": ["Change color to {color}", "Switch color to {color}", "Set color to {color}"],
        "values": ["red", "blue", "green"],
        "slot_name": "color"
    },
    "TEMPERATURE": {
        "phrases": ["Temperature {degrees}", "Change the temperature to {degrees}", "Set the temperature to {degrees}"],
        "values": ["18 degrees", "22 degrees", "26 degrees"],
        "slot_name": "degrees"
    },
    "TIMER": {
        "phrases": ["Timer {duration}", "Countdown for {duration}", "Start a timer for {duration}"],
        "values": ["10 seconds", "30 seconds", "1 minute"],
        "slot_name": "duration"
    },
    "ALARM": {
        "phrases": ["Alarm {time}", "Wake me up at {time}", "Set an alarm for {time}"],
        "values": ["6 AM", "8 AM", "9 PM"],
        "slot_name": "time"
    },
    "CREATE_REMINDER": {
        "phrases": ["Reminder {task}", "Remind me to {task}", "Create a reminder to {task}"],
        "values": ["drink water", "study", "exercise"],
        "slot_name": "task"
    }
}

def record_audio(filepath, duration=DURATION, sr=SAMPLE_RATE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"\n[!] Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=CHANNELS, dtype='int16')
    sd.wait()
    print("[*] Recording complete.")

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

def generate_tasks(speaker_id):
    tasks = []

    # 1. Fixed intents
    for intent, variations in FIXED_PHRASES.items():
        for i, phrase in enumerate(variations):
            v_id = f"v{i+1}"
            filename = f"{intent}_{speaker_id}_{v_id}.wav"
            rel_path = os.path.join(intent, filename)
            tasks.append({
                "path": rel_path,
                "label": intent,
                "intent": intent,
                "speaker": speaker_id,
                "split": "train",
                "phrase_id": v_id,
                "transcript": phrase,
                "slot": "none",
                "slot_value": "none",
                "duration_sec": DURATION
            })

    # 2. Slotted intents
    for intent, config in SLOTTED_PHRASES.items():
        for i, phrase_template in enumerate(config["phrases"]):
            v_id = f"v{i+1}"
            for val in config["values"]:
                phrase = phrase_template.replace("{" + config["slot_name"] + "}", val)
                formatted_val = val.replace(" ", "_").replace(":", "_")
                label = f"{intent}_{formatted_val}"
                filename = f"{label}_{speaker_id}_{v_id}.wav"
                rel_path = os.path.join(label, filename)
                tasks.append({
                    "path": rel_path,
                    "label": label,
                    "intent": intent,
                    "speaker": speaker_id,
                    "split": "train",
                    "phrase_id": v_id,
                    "transcript": phrase,
                    "slot": config["slot_name"],
                    "slot_value": val,
                    "duration_sec": DURATION
                })

    return tasks

def save_manifest(tasks, out_dir):
    import csv
    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = ["path", "label", "intent", "speaker", "split", "phrase_id", "transcript", "slot", "slot_value", "duration_sec"]
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tasks:
            writer.writerow(t)
    print(f"\n[+] Manifest saved to {manifest_path}")

def package_zip(out_dir, speaker_id):
    zip_name = f"custom_voice_{speaker_id}.zip"
    import glob
    
    # Check out_dir first
    wav_files = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
    target_scan_dir = out_dir

    # Fallback: check finetune_16k if out_dir is empty
    if not wav_files:
        alt_dirs = ["finetune_16k", "../finetune_16k"]
        for alt in alt_dirs:
            if os.path.exists(alt):
                alt_wavs = glob.glob(os.path.join(alt, "**", "*.wav"), recursive=True)
                if alt_wavs:
                    print(f"[*] Detected {len(alt_wavs)} audio file(s) in '{alt}'. Packaging from '{alt}'...")
                    target_scan_dir = alt
                    wav_files = alt_wavs
                    break

    if not wav_files:
        print(f"[!] WARNING: No .wav audio files found in '{out_dir}' or 'finetune_16k'.")
        print("    Did you record under a different Speaker ID or directory?")
        return

    print(f"[*] Packaging {len(wav_files)} recordings from '{target_scan_dir}' into {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(target_scan_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=target_scan_dir)
                zf.write(abs_path, arcname=os.path.join(speaker_id, rel_path))

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"[+] Successfully created {zip_name} ({size_mb:.2f} MB, {len(wav_files)} audio files)!")
    print(f"    Transfer {zip_name} to DGX and unzip into 'data/custom_voices/'!")

def run_sequential_recording(tasks, out_dir, speaker_id):
    recorded = []
    total = len(tasks)
    print(f"\nStarting recording session for speaker '{speaker_id}' ({total} phrases total).")

    for idx, task in enumerate(tasks):
        abs_path = os.path.join(out_dir, task['path'])
        already_done = os.path.exists(abs_path)
        status_tag = "[RECORDED]" if already_done else "[PENDING]"

        print(f"\n--- [{idx+1}/{total}] {status_tag} Intent: {task['intent']} ---")
        if task['slot'] != "none":
            print(f"Slot: {task['slot']} -> '{task['slot_value']}'")
        print(f"Phrase: \"{task['transcript']}\"")

        while True:
            prompt = "Press Enter to record (or 's' to skip, 'q' to quit): "
            if already_done:
                prompt = "Press Enter to RE-RECORD (or 's' to keep existing, 'q' to quit): "
            choice = input(prompt).strip().lower()
            if choice == 'q':
                save_manifest(tasks, out_dir)
                package_zip(out_dir, speaker_id)
                return
            elif choice == 's':
                break

            record_audio(abs_path)
            recorded.append(task)

            redo = input("Press Enter to accept, or 'r' to re-record: ").strip().lower()
            if redo != 'r':
                break

    save_manifest(tasks, out_dir)
    package_zip(out_dir, speaker_id)
    print("\n[+] Sequential recording completed!")

def run_specific_recording(tasks, out_dir, speaker_id):
    while True:
        print("\n--- Search & Re-record a Specific Phrase ---")
        query = input("Enter search term (intent, slot, phrase, or 'q' to back): ").strip().lower()
        if query == 'q':
            break

        matches = [
            (i, t) for i, t in enumerate(tasks)
            if query in t['intent'].lower()
            or query in t['transcript'].lower()
            or query in t['slot_value'].lower()
        ]

        if not matches:
            print("[!] No phrases matched your search.")
            continue

        for i, t in matches:
            exists = "✓" if os.path.exists(os.path.join(out_dir, t['path'])) else " "
            print(f"[{i}] [{exists}] {t['intent']} ({t['slot_value']}) -> \"{t['transcript']}\"")

        sel = input("Enter the number to re-record (or Enter to cancel): ").strip()
        if sel.isdigit() and int(sel) < len(tasks):
            t = tasks[int(sel)]
            abs_path = os.path.join(out_dir, t['path'])
            print(f"\nRecording: \"{t['transcript']}\"")
            record_audio(abs_path)
            print("[+] Saved!")
            save_manifest(tasks, out_dir)

def main():
    print("=" * 60)
    print(" Voice Control Model - Multi-Speaker Voice Recorder ")
    print("=" * 60)

    speaker_id = input("Enter Speaker ID (e.g. s_user, s_friend1) [Default: s_user]: ").strip()
    if not speaker_id:
        speaker_id = "s_user"

    out_dir = os.path.join("data", "custom_voices", speaker_id)
    os.makedirs(out_dir, exist_ok=True)

    tasks = generate_tasks(speaker_id)

    while True:
        print(f"\nSpeaker: {speaker_id} | Working Directory: {out_dir}")
        print("1. Record all phrases sequentially")
        print("2. Search and re-record a specific phrase")
        print("3. Package recordings to ZIP archive")
        print("4. Switch Speaker ID")
        print("5. Quit")

        choice = input("Select an option (1-5): ").strip()
        if choice == '1':
            run_sequential_recording(tasks, out_dir, speaker_id)
        elif choice == '2':
            run_specific_recording(tasks, out_dir, speaker_id)
        elif choice == '3':
            save_manifest(tasks, out_dir)
            package_zip(out_dir, speaker_id)
        elif choice == '4':
            speaker_id = input("Enter new Speaker ID: ").strip() or "s_user"
            out_dir = os.path.join("data", "custom_voices", speaker_id)
            os.makedirs(out_dir, exist_ok=True)
            tasks = generate_tasks(speaker_id)
        elif choice in ('5', 'q'):
            break

if __name__ == "__main__":
    main()

