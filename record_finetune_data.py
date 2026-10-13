import os
import time
import json
import wave
import pandas as pd

# Fallback in case sounddevice isn't available
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    exit(1)

SAMPLE_RATE = 16000
DURATION = 2.5
CHANNELS = 1

phrase_variations = {
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

slotted_variations = {
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

def record_audio(filename, duration=DURATION, sr=SAMPLE_RATE):
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"\n[!] Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=CHANNELS, dtype='int16')
    sd.wait()
    print("[*] Recording complete.")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

def generate_tasks(speaker_id):
    tasks = []
    
    # 1. Fixed Intents
    for intent, variations in phrase_variations.items():
        label_dir = f"{intent}_16k"
        for i, phrase in enumerate(variations):
            v_id = f"v{i+1}"
            rel_path = os.path.join(label_dir, f"{intent}_{speaker_id}_{v_id}_clean.wav")
            tasks.append({
                "path": rel_path,
                "label": intent,
                "intent": intent,
                "speaker": speaker_id,
                "split": "train",
                "phrase_id": v_id,
                "variant_id": "clean",
                "transcript": phrase,
                "slot": "none",
                "slot_value": "none",
                "duration_sec": DURATION
            })

    # 2. Slotted Intents
    for intent, config in slotted_variations.items():
        for i, phrase_template in enumerate(config["phrases"]):
            v_id = f"v{i+1}"
            for val in config["values"]:
                phrase = phrase_template.replace("{" + config["slot_name"] + "}", val)
                formatted_val = val.replace(" ", "_")
                if "AM" in formatted_val or "PM" in formatted_val:
                    formatted_val = formatted_val.replace(":", "_")
                
                label = f"{intent}_{formatted_val}"
                label_dir = f"{label}_16k"
                rel_path = os.path.join(label_dir, f"{label}_{speaker_id}_{v_id}_clean.wav")
                
                tasks.append({
                    "path": rel_path,
                    "label": label,
                    "intent": intent,
                    "speaker": speaker_id,
                    "split": "train",
                    "phrase_id": v_id,
                    "variant_id": "clean",
                    "transcript": phrase,
                    "slot": config["slot_name"],
                    "slot_value": val,
                    "duration_sec": DURATION
                })
    return tasks

def save_manifest(new_records, out_dir):
    if not new_records:
        return
    manifest_path = os.path.join(out_dir, "manifest_16k.csv")
    df_new = pd.DataFrame(new_records)
    
    if os.path.exists(manifest_path):
        df_old = pd.read_csv(manifest_path)
        # Combine and drop duplicates based on file path to avoid duplicate manifest entries
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.drop_duplicates(subset=['path'], keep='last', inplace=True)
    else:
        df = df_new
        
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest updated at {manifest_path}")

def run_sequential_recording(tasks, out_dir):
    records = []
    for idx, task in enumerate(tasks):
        print(f"\n--- [{idx+1}/{len(tasks)}] Intent: {task['intent']} ---")
        print(f"Phrase to read: '{task['transcript']}'")
        
        while True:
            user_input = input("Press Enter to record (or 'q' to quit, 's' to skip): ")
            if user_input.lower() == 'q':
                save_manifest(records, out_dir)
                return
            elif user_input.lower() == 's':
                break
                
            abs_path = os.path.join(out_dir, task['path'])
            record_audio(abs_path)
            records.append(task)
            
            # Allow immediately re-recording if the user wasn't happy
            redo = input("Press Enter to accept, or 'r' to re-record this phrase: ")
            if redo.lower() != 'r':
                break
                
    save_manifest(records, out_dir)
    print("\nSequential recording finished!")

def run_specific_recording(tasks, out_dir):
    records = []
    while True:
        print("\n--- Re-Record Specific Audio ---")
        search = input("Enter part of the phrase, intent, or filename to search (or 'q' to go back): ")
        if search.lower() == 'q':
            break
            
        matches = [
            (idx, t) for idx, t in enumerate(tasks) 
            if search.lower() in t['intent'].lower() 
            or search.lower() in t['transcript'].lower() 
            or search.lower() in os.path.basename(t['path']).lower()
        ]
        
        if not matches:
            print("No matching phrases found.")
            continue
            
        print("\nMatches:")
        for idx, t in matches:
            print(f"[{idx}] {t['intent']} - '{t['transcript']}' -> {os.path.basename(t['path'])}")
            
        sel = input("\nEnter the bracketed index number to re-record (or Enter to cancel): ")
        if sel.isdigit():
            idx = int(sel)
            if idx < len(tasks):
                task = tasks[idx]
                print(f"\n--- Intent: {task['intent']} ---")
                print(f"Phrase to read: '{task['transcript']}'")
                
                user_input = input("Press Enter to record (or 'q' to cancel): ")
                if user_input.lower() != 'q':
                    abs_path = os.path.join(out_dir, task['path'])
                    record_audio(abs_path)
                    records.append(task)
                    print(f"Successfully re-recorded: {os.path.basename(task['path'])}")
            else:
                print("Invalid index.")
                
    save_manifest(records, out_dir)

def main():
    print("=== Fine-tuning Data Recorder ===")
    speaker_id = input("Enter your speaker ID (e.g., s_user) [Default: s_user]: ").strip()
    if not speaker_id:
        speaker_id = "s_user"
        
    out_dir = "finetune_16k"
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate list of all phrases
    tasks = generate_tasks(speaker_id)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Record all phrases sequentially")
        print("2. Search and re-record a specific phrase")
        print("3. Quit")
        
        choice = input("Select an option (1-3): ").strip()
        if choice == '1':
            run_sequential_recording(tasks, out_dir)
        elif choice == '2':
            run_specific_recording(tasks, out_dir)
        elif choice == '3' or choice.lower() == 'q':
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()

