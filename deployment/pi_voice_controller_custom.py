#!/usr/bin/env python3
"""
Raspberry Pi 5 Voice-Controlled Assistant & Smart Display Integration
Includes real-time 44.1kHz to 16kHz audio resampling for USB microphones.
Auto-pause logic to silence music while listening to a command.
Asynchronous non-blocking dispatch to the local web dashboard (http://127.0.0.1:5000).
"""

import ctypes
import os
import sys

# Disable all Windows Error popups (e.g., missing DLLs) which silently deadlock the script
if sys.platform == "win32":
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

import json
import time
import threading
import traceback
import urllib.request
import numpy as np
import sounddevice as sd

from gpiozero import PWMLED
import pygame

import openwakeword
from openwakeword.model import Model
import onnxruntime as ort

# --- Critical Fix for Librosa / Soundfile Deadlock ---
from unittest.mock import MagicMock
sys.modules['soundfile'] = MagicMock()

import librosa

# =====================================================================
# 1. CONFIGURATION & ADJUSTABLE PARAMETERS
# =====================================================================

# Web Dashboard Synchronization Endpoint
WEB_SERVER_URL = "http://127.0.0.1:5000/api/command"

# Microphone Hardware Settings (Based on Diagnostic)
MIC_DEVICE_ID = 1            # Your Fifine USB Audio ID
MIC_SAMPLE_RATE = 44100      # Native microphone sample rate
MIC_CHANNELS = 1             # Requesting mono (OS will downmix from 2)
DTYPE = 'float32'

# AI Model Audio Requirements
MODEL_SAMPLE_RATE = 16000    # Required by openwakeword and ONNX model
BUFFER_DURATION = 3.0        # Seconds of rolling audio kept in memory
BUFFER_SIZE = int(MODEL_SAMPLE_RATE * BUFFER_DURATION)

# Wake Word Settings (80ms chunks)
MIC_BLOCKSIZE = int(MIC_SAMPLE_RATE * 0.08)  # 3528 samples per chunk from mic
WAKE_WORD_DURATION = 1.0
WAKE_WORD_SAMPLES = int(MODEL_SAMPLE_RATE * WAKE_WORD_DURATION)
WAKE_WORD_THRESHOLD = 0.04
WAKE_WORD_INTERVAL = 0.08  # 80ms (exactly 1280 samples at 16kHz)
WAKE_WORD_SOFTWARE_GAIN = 2.0  # Multiplier to boost faint voices before wake word detection

# Command Recording Settings
COMMAND_DURATION = 2.5
COMMAND_SAMPLES = int(MODEL_SAMPLE_RATE * COMMAND_DURATION)
INTENT_CONFIDENCE_THRESHOLD = 30.0

# Paths to your dataset label and model files
LABELS_PATH = "intents.json"
SLOTS_PATH = "slots.json"
ONNX_MODEL_PATH = "vcm_int8.onnx"

# Raspberry Pi 5 Hardware Configuration (BCM Pin Numbers)
RED_PIN = 17          # Physical Pin 11
GREEN_PIN = 27        # Physical Pin 13
BLUE_PIN = 22         # Physical Pin 15
PWM_FREQUENCY = 100   # 100 Hz PWM for smooth, flicker-free dimming

# Music Player Configuration
MUSIC_FILES = ["music1.mp3", "music2.mp3"]
INITIAL_VOLUME = 0.8

# State Machine Constants
STATE_LISTENING = 0
STATE_RECORDING = 1
STATE_INFERENCE = 2


# =====================================================================
# 2. HELPER FUNCTIONS & THREAD-SAFE RING BUFFER
# =====================================================================

def send_web_command_async(intent, slot="", confidence=0.0):
    """
    Sends detected intent and slot to the local web server dashboard.
    Runs asynchronously in a background daemon thread so it never delays the audio loop.
    """
    def _worker():
        try:
            payload = json.dumps({
                "intent": intent,
                "slot": slot,
                "confidence": float(confidence)
            }).encode('utf-8')
            req = urllib.request.Request(
                WEB_SERVER_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=0.6) as _:
                pass
        except Exception:
            # Silent fallback if web server is not running
            pass

    threading.Thread(target=_worker, daemon=True).start()


def load_label_mappings(labels_path, slots_path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        intents = json.load(f)
    intent2id = {intent: idx for idx, intent in enumerate(intents)}
    id2intent = {idx: intent for intent, idx in intent2id.items()}

    with open(slots_path, 'r', encoding='utf-8') as f:
        slots_data = json.load(f)

    slot2id = {"No Slot": 0}
    slot_idx = 1
    for intent, slot_info in slots_data.items():
        for value in slot_info['values']:
            if value not in slot2id:
                slot2id[value] = slot_idx
                slot_idx += 1

    id2slot = {idx: value for value, idx in slot2id.items()}
    return id2intent, id2slot, slots_data


def compute_confidence(logits):
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    return float(np.max(probabilities) * 100.0)


class AudioBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.lock = threading.RLock()
        self.write_index = 0
        self.total_written = 0

    def write(self, data):
        with self.lock:
            n = len(data)
            if n >= self.size:
                self.buffer[:] = data[-self.size:]
            else:
                end_idx = self.write_index + n
                if end_idx <= self.size:
                    self.buffer[self.write_index:end_idx] = data
                else:
                    part1_len = self.size - self.write_index
                    self.buffer[self.write_index:] = data[:part1_len]
                    self.buffer[:end_idx % self.size] = data[part1_len:]
            self.write_index = (self.write_index + n) % self.size
            self.total_written += n

    def read_since(self, read_index):
        with self.lock:
            available = self.total_written - read_index
            if available <= 0:
                return np.zeros(0, dtype=np.float32), self.total_written
            if available > self.size:
                available = self.size
                read_index = self.total_written - self.size

            out = np.zeros(available, dtype=np.float32)
            start_idx = read_index % self.size
            if start_idx + available <= self.size:
                out[:] = self.buffer[start_idx:start_idx + available]
            else:
                part1_len = self.size - start_idx
                out[:part1_len] = self.buffer[start_idx:]
                out[part1_len:] = self.buffer[:available - part1_len]
            return out, self.total_written

    def read_latest(self, num_samples):
        with self.lock:
            if num_samples > self.size:
                num_samples = self.size

            out = np.zeros(num_samples, dtype=np.float32)
            start_idx = (self.write_index - num_samples) % self.size

            if start_idx < self.write_index:
                out[:] = self.buffer[start_idx:self.write_index]
            else:
                part1_len = self.size - start_idx
                out[:part1_len] = self.buffer[start_idx:]
                out[part1_len:] = self.buffer[:self.write_index]
            return out


# =====================================================================
# 3. MAIN APPLICATION LOOP
# =====================================================================

def main():
    print("Initializing LEDs...")
    try:
        leds = {
            "red": PWMLED(RED_PIN, frequency=PWM_FREQUENCY),
            "green": PWMLED(GREEN_PIN, frequency=PWM_FREQUENCY),
            "blue": PWMLED(BLUE_PIN, frequency=PWM_FREQUENCY),
        }
        for led in leds.values():
            led.off()
    except Exception as e:
        print(f"[Error] Failed to init LEDs: {e}")
        leds = None

    print("Initializing PyGame Audio...")
    audio_ready = False
    music_paused = False
    current_music_index = 0
    available_music = [f for f in MUSIC_FILES if os.path.isfile(f)]
    
    if available_music:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(available_music[current_music_index])
            pygame.mixer.music.set_volume(INITIAL_VOLUME)
            audio_ready = True
            music_paused = False
        except Exception as e:
            print(f"[Warning] Could not initialize music: {e}")
    else:
        print(f"[Warning] No music files found from {MUSIC_FILES}.")

    print("Loading label and slot mappings...")
    try:
        id2intent, id2slot, slots_data = load_label_mappings(LABELS_PATH, SLOTS_PATH)
        print(f"Loaded {len(id2intent)} intents and {len(id2slot)} slot values.")
    except Exception as e:
        print(f"Warning: Could not load label mappings. Error: {e}")
        id2intent, id2slot, slots_data = {}, {}, {}

    print("Loading custom wake word model (betelgeuse)...")
    model_path = os.path.join(os.path.dirname(__file__), "betelgeuse.onnx")
    oww_model = Model(wakeword_models=[model_path], inference_framework='onnx')

    print("Loading ONNX inference session...")
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load '{ONNX_MODEL_PATH}'. Error: {e}")
        ort_session = None

    print("Pre-warming librosa audio feature extractor and resampler...")
    dummy_audio = np.zeros(COMMAND_SAMPLES, dtype=np.float32)
    _ = librosa.resample(dummy_audio[:4000], orig_sr=MIC_SAMPLE_RATE, target_sr=MODEL_SAMPLE_RATE)
    
    dummy_mel = librosa.feature.melspectrogram(
        y=dummy_audio, sr=MODEL_SAMPLE_RATE, n_fft=512, hop_length=160,
        win_length=512, n_mels=64, htk=True, norm='slaney'
    )
    _ = librosa.power_to_db(dummy_mel, ref=1.0)

    audio_buffer = AudioBuffer(BUFFER_SIZE)

    # Real-time resampling callback
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio stream status: {status}")
        
        # Convert incoming 44.1kHz audio to 16kHz required by the models
        audio_16k = librosa.resample(
            indata[:, 0], 
            orig_sr=MIC_SAMPLE_RATE, 
            target_sr=MODEL_SAMPLE_RATE
        )
        audio_buffer.write(audio_16k)

    # Initialize stream with specific Device ID and 44.1kHz sampling rate
    stream = sd.InputStream(
        device=MIC_DEVICE_ID,
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
        blocksize=MIC_BLOCKSIZE 
    )

    stream.start()
    print("Started listening for wake word ('Betelgeuse')...")

    state = STATE_LISTENING
    last_wake_check_time = time.time()
    recording_start_samples = 0
    was_playing_before_wake = False  # Track music state across loops

    def log_debug(msg):
        with open("debug.log", "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f}: {msg}\n")

    try:
        oww_read_index = 0

        while True:
            try:
                current_time = time.time()

                if state == STATE_LISTENING:
                    if current_time - last_wake_check_time >= WAKE_WORD_INTERVAL:
                        last_wake_check_time = current_time

                        recent_audio, oww_read_index = audio_buffer.read_since(oww_read_index)

                        if recent_audio is not None and len(recent_audio) > 0:
                            # Apply software gain to boost faint voices, clip to avoid distortion, and convert to int16
                            audio_int16 = np.clip(recent_audio * 32767 * WAKE_WORD_SOFTWARE_GAIN, -32768, 32767).astype(np.int16)
                            prediction = oww_model.predict(audio_int16)
                            # Extract the confidence score dynamically for our custom model
                            confidence = list(prediction.values())[0] if prediction else 0.0

                            if confidence > 0.05:
                                log_debug(f"Confidence spiked: {confidence:.2f}")
                                # Print occasional confidence spikes to help user debug
                                print(f"  [Debug] Wake word confidence: {confidence:.2f}% (Threshold: {WAKE_WORD_THRESHOLD})")
                            
                            # Check for completely silent audio (mic issue)
                            rms_volume = np.sqrt(np.mean(recent_audio**2))
                            if rms_volume < 1e-6 and current_time - getattr(main, 'last_silent_warn', 0) > 5.0:
                                print("\n[WARNING] Microphone audio is completely silent. Check your default mic settings in Windows!")
                                main.last_silent_warn = current_time

                            if confidence > WAKE_WORD_THRESHOLD:
                                print(f"\n[WAKE DETECTED] Listening for command... (Confidence: {confidence:.2f})")
                                log_debug("WAKE DETECTED, setting state=STATE_RECORDING")
                                
                                # Auto-Pause music while listening
                                was_playing_before_wake = False
                                if audio_ready and pygame.mixer.music.get_busy():
                                    pygame.mixer.music.pause()
                                    was_playing_before_wake = True
                                    print("   [Auto-Pausing music to listen]")

                                with audio_buffer.lock:
                                    recording_start_samples = audio_buffer.total_written
                                state = STATE_RECORDING

                elif state == STATE_RECORDING:
                    with audio_buffer.lock:
                        samples_since_wake = audio_buffer.total_written - recording_start_samples

                    if samples_since_wake >= COMMAND_SAMPLES:
                        log_debug(f"Captured {samples_since_wake} samples, transitioning to STATE_INFERENCE")
                        state = STATE_INFERENCE

                elif state == STATE_INFERENCE:
                    inference_start_time = time.perf_counter()
                    log_debug("Entering STATE_INFERENCE")
                    command_audio = audio_buffer.read_latest(COMMAND_SAMPLES)
                    audio_length_ms = (len(command_audio) / MODEL_SAMPLE_RATE) * 1000

                    if ort_session is not None:
                        log_debug("Computing melspectrogram")
                        
                        # Normalize command audio to handle faint voices
                        max_amp = np.max(np.abs(command_audio))
                        if max_amp > 0.005:  # Avoid amplifying pure silence
                            command_audio = command_audio / max_amp

                        mel_spectrogram = librosa.feature.melspectrogram(
                            y=command_audio,
                            sr=MODEL_SAMPLE_RATE,
                            n_fft=512,
                            hop_length=160,
                            win_length=512,
                            center=True,
                            pad_mode='reflect',
                            n_mels=64,
                            htk=True,          
                            norm='slaney'      
                        )
                        log_mel = librosa.power_to_db(mel_spectrogram, ref=1.0, top_db=80.0)

                        target_length = 251
                        if log_mel.shape[1] > target_length:
                            log_mel = log_mel[:, :target_length]
                        elif log_mel.shape[1] < target_length:
                            pad_width = target_length - log_mel.shape[1]
                            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')

                        input_tensor = log_mel.reshape(1, 1, 64, 251).astype(np.float32)

                        inputs = {ort_session.get_inputs()[0].name: input_tensor}
                        outputs = ort_session.run(None, inputs)

                        intent_logits = outputs[0]
                        slot_logits = outputs[1]

                        intent_id = int(np.argmax(intent_logits))
                        slot_id = int(np.argmax(slot_logits))

                        intent_conf = compute_confidence(intent_logits)
                        slot_conf = compute_confidence(slot_logits)

                        if intent_conf < INTENT_CONFIDENCE_THRESHOLD:
                            intent_label = "UNKNOWN"
                        else:
                            intent_label = id2intent.get(intent_id, f"Unknown Intent ({intent_id})")
                        slot_label = id2slot.get(slot_id, f"Unknown Slot ({slot_id})")

                        slot_type = slots_data.get(intent_label, {}).get("slot", None)

                        inference_end_time = time.perf_counter()
                        infer_ms = (inference_end_time - inference_start_time) * 1000

                        if slot_type and slot_label != "No Slot":
                            print(
                                f"[DETECTED] Intent: {intent_label} ({intent_conf:.1f}%) | "
                                f"Slot ({slot_type}): {slot_label} ({slot_conf:.1f}%) "
                                f"[IDs: intent={intent_id}, slot={slot_id}]"
                            )
                        else:
                            print(
                                f"[DETECTED] Intent: {intent_label} ({intent_conf:.1f}%) | "
                                f"Slot: {slot_label} ({slot_conf:.1f}%) "
                                f"[IDs: intent={intent_id}, slot={slot_id}]"
                            )
                        print(f"infer_ms={infer_ms:.0f} audio_ms={audio_length_ms:.0f}")

                        # --- Dispatches Intent Event to Local Web Dashboard ---
                        send_web_command_async(intent_label, slot_label, intent_conf)

                        # =====================================================
                        # INTENT EXECUTION LOGIC (All 19 Dataset Intents)
                        # =====================================================

                        # --- Lighting Intents ---
                        if intent_label == "LIGHT_ON" and leds:
                            print("-> Turning lights ON")
                            for led in leds.values():
                                led.on()

                        elif intent_label == "LIGHT_OFF" and leds:
                            print("-> Turning lights OFF")
                            for led in leds.values():
                                led.off()

                        elif intent_label == "COLOR" and leds:
                            color = slot_label.lower().strip()
                            if color in leds:
                                print(f"-> Setting color to {color}")
                                for name, led in leds.items():
                                    if name == color:
                                        led.on()
                                    else:
                                        led.off()
                            else:
                                print(f"-> Unknown color slot: '{color}'")

                        elif intent_label == "BRIGHTNESS" and leds:
                            try:
                                pct_str = slot_label.replace("%", "").split()[0]
                                pct = float(pct_str) / 100.0
                                pct = max(0.0, min(1.0, pct))  
                                print(f"-> Setting brightness to {pct_str}%")
                                any_lit = any(led.is_lit for led in leds.values())
                                for led in leds.values():
                                    if any_lit:
                                        if led.is_lit:
                                            led.value = pct
                                    else:
                                        led.value = pct
                            except Exception as e:
                                print(f"Failed to set brightness from slot '{slot_label}': {e}")

                        # --- Music Control Intents ---
                        elif intent_label == "PLAY_MUSIC" and audio_ready:
                            if music_paused:
                                print("-> Resuming music")
                                pygame.mixer.music.unpause()
                                music_paused = False
                            else:
                                print("-> Playing music")
                                pygame.mixer.music.play()
                                music_paused = False

                        elif intent_label == "PAUSE" and audio_ready:
                            print("-> Pausing music")
                            pygame.mixer.music.pause()
                            music_paused = True

                        elif intent_label == "STOP" and audio_ready:
                            print("-> Stopping music")
                            pygame.mixer.music.stop()
                            music_paused = False

                        elif intent_label == "NEXT" and audio_ready:
                            if available_music:
                                current_music_index = (current_music_index + 1) % len(available_music)
                                next_song = available_music[current_music_index]
                                print(f"-> Playing next song: {next_song}")
                                try:
                                    pygame.mixer.music.load(next_song)
                                    pygame.mixer.music.play()
                                    music_paused = False
                                except Exception as e:
                                    print(f"[Error] Failed to load {next_song}: {e}")
                            else:
                                print("-> Rewinding music (Next track not supported)")
                                pygame.mixer.music.rewind()
                                music_paused = False

                        elif intent_label == "VOLUME_UP" and audio_ready:
                            vol = min(1.0, pygame.mixer.music.get_volume() + 0.2)
                            print(f"-> Volume UP ({vol:.1f})")
                            pygame.mixer.music.set_volume(vol)

                        elif intent_label == "VOLUME_DOWN" and audio_ready:
                            vol = max(0.0, pygame.mixer.music.get_volume() - 0.2)
                            print(f"-> Volume DOWN ({vol:.1f})")
                            pygame.mixer.music.set_volume(vol)

                        # --- Climate Control ---
                        elif intent_label == "TEMPERATURE":
                            print(f"-> Adjusting thermostat to {slot_label}")

                        # --- Information Intents ---
                        elif intent_label == "WEATHER":
                            print("-> Refreshing and presenting live weather information")

                        elif intent_label == "TIME":
                            now_time_str = time.strftime("%I:%M:%S %p")
                            print(f"-> Current local time: {now_time_str}")

                        # --- Timers & Alarms ---
                        elif intent_label == "TIMER":
                            print(f"-> Set active countdown timer for {slot_label}")

                        elif intent_label == "ALARM":
                            print(f"-> Scheduled alarm for {slot_label}")

                        # --- Communication Simulation ---
                        elif intent_label == "CALL":
                            print("-> Initiating simulated phone call")

                        elif intent_label == "MESSAGE":
                            print("-> Sending simulated outgoing text message")

                        # --- Reminders ---
                        elif intent_label == "CREATE_REMINDER":
                            print(f"-> Added new reminder: '{slot_label}'")

                        elif intent_label == "LIST_REMINDERS":
                            print("-> Displaying current reminder tasks")
                            
                        # --- Auto-Resume Music Logic ---
                        if was_playing_before_wake:
                            # Do not auto-resume if the user specifically asked to pause or stop
                            if intent_label not in ["PAUSE", "STOP"]:
                                pygame.mixer.music.unpause()
                                music_paused = False
                                print("   [Auto-Resuming music]")
                            else:
                                music_paused = True

                    else:
                        print("[ERROR] ONNX Session is unavailable.")

                    log_debug("Resetting openwakeword")
                    oww_model.reset()

                    with audio_buffer.lock:
                        oww_read_index = audio_buffer.total_written

                    state = STATE_LISTENING
                    last_wake_check_time = time.time()
                    log_debug("Finished STATE_INFERENCE, returning to STATE_LISTENING")

                time.sleep(0.01)

            except Exception as e:
                print(f"Error during state machine processing: {e}")
                log_debug(f"Exception: {e}")
                traceback.print_exc()
                state = STATE_LISTENING
                last_wake_check_time = time.time()
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        print("Cleaning up hardware...")
        if 'stream' in locals() and stream:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass
        if 'leds' in locals() and leds:
            for led in leds.values():
                try:
                    led.off()
                    led.close()
                except Exception:
                    pass
        if 'audio_ready' in locals() and audio_ready:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
