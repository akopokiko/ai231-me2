#!/usr/bin/env python3
"""
Cross-Platform Voice Controller for Windows Laptop and Raspberry Pi 5.
Features:
- Automatic Hardware Abstraction Layer (HAL): Auto-mocks GPIO on PC/laptop and uses real GPIO on RPi 5.
- Audio cues (voice_start.ogg, voice_success.ogg, voice_unknown.ogg) with non-interfering playback.
- Offline local voice inference: OpenWakeWord ('Betelgeuse') + ONNX BC-ResNet.
- Free open-source online integration for WEATHER (Open-Meteo API) with offline fallback.
- Real-time synchronization with local web dashboard (http://127.0.0.1:5000).
"""

import os
import sys
import time
import json
import threading
import argparse
import urllib.request
import urllib.parse
import http.client
import numpy as np
import sounddevice as sd
import pygame
import onnxruntime as ort

# =====================================================================
# 1. HARDWARE ABSTRACTION LAYER (HAL)
# =====================================================================

def is_raspberry_pi():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            if 'raspberry pi' in model:
                return True
    except Exception:
        pass
    return False

class MockPWMLED:
    def __init__(self, color_name, update_callback):
        self.color_name = color_name
        self.update_callback = update_callback
        self._value = 0.0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = max(0.0, min(1.0, float(val)))
        self.update_callback()

    def on(self):
        self.value = 1.0

    def off(self):
        self.value = 0.0

    @property
    def is_lit(self):
        return self._value > 0.01

    def close(self):
        pass

class LEDController:
    def __init__(self, force_mock=False, red_pin=17, green_pin=27, blue_pin=22):
        self.use_mock = force_mock or (not is_raspberry_pi())
        self.pins = {"red": red_pin, "green": green_pin, "blue": blue_pin}
        self.leds = {}

        if self.use_mock:
            print("[HAL] Using Mock GPIO LED Controller (Laptop / PC mode).")
            for color in ["red", "green", "blue"]:
                self.leds[color] = MockPWMLED(color, self._on_led_change)
        else:
            print("[HAL] Detected Raspberry Pi 5. Initializing physical GPIOzero PWMLEDs...")
            try:
                from gpiozero import PWMLED
                for color, pin in self.pins.items():
                    self.leds[color] = PWMLED(pin, frequency=100)
            except Exception as e:
                print(f"[!] Warning: Failed to initialize physical GPIO ({e}). Falling back to Mock LEDs.")
                self.use_mock = True
                for color in ["red", "green", "blue"]:
                    self.leds[color] = MockPWMLED(color, self._on_led_change)

        self.off_all()

    def _on_led_change(self):
        # Broadcast LED values to web server
        send_web_event_async("api/leds", {
            "red": self.leds["red"].value,
            "green": self.leds["green"].value,
            "blue": self.leds["blue"].value
        })

    def on_all(self):
        for led in self.leds.values():
            led.on()
        self._on_led_change()

    def off_all(self):
        for led in self.leds.values():
            led.off()
        self._on_led_change()

    def set_color(self, target_color):
        target = target_color.lower().strip()
        for color, led in self.leds.items():
            if color == target:
                led.on()
            else:
                led.off()
        self._on_led_change()

    def set_brightness(self, fraction):
        fraction = max(0.0, min(1.0, float(fraction)))
        any_lit = any(led.is_lit for led in self.leds.values())
        for led in self.leds.values():
            if any_lit:
                if led.is_lit:
                    led.value = fraction
            else:
                led.value = fraction
        self._on_led_change()

    def close(self):
        for led in self.leds.values():
            led.off()
            led.close()

# =====================================================================
# 2. WEB SYNCHRONIZATION HELPERS
# =====================================================================

WEB_SERVER_BASE = "http://127.0.0.1:5000"

def send_web_event_async(endpoint, data):
    def _worker():
        try:
            url = f"{WEB_SERVER_BASE}/{endpoint}"
            payload = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=0.5) as _:
                pass
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()

class AudioBroadcaster:
    """High-frequency audio level and state streamer for Web UI visualization."""
    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.level = 0.0
        self.state = "LISTENING"
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_level(self, level):
        with self.lock:
            self.level = float(level)

    def set_state(self, state):
        with self.lock:
            self.state = str(state)

    def stop(self):
        self.running = False

    def _loop(self):
        conn = None
        while self.running:
            try:
                with self.lock:
                    lvl = self.level
                    st = self.state

                payload = json.dumps({"level": lvl, "state": st})
                if conn is None:
                    conn = http.client.HTTPConnection(self.host, self.port, timeout=0.2)

                headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
                conn.request("POST", "/api/audio", body=payload, headers=headers)
                resp = conn.getresponse()
                resp.read()
            except Exception:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(0.1)
            time.sleep(0.05)  # 20 Hz update rate

# =====================================================================
# 3. THREAD-SAFE AUDIO RING BUFFER
# =====================================================================

class AudioRingBuffer:
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
                    p1 = self.size - self.write_index
                    self.buffer[self.write_index:] = data[:p1]
                    self.buffer[:end_idx % self.size] = data[p1:]
            self.write_index = (self.write_index + n) % self.size
            self.total_written += n

    def read_since(self, read_index):
        with self.lock:
            avail = self.total_written - read_index
            if avail <= 0:
                return np.zeros(0, dtype=np.float32), self.total_written
            if avail > self.size:
                avail = self.size
                read_index = self.total_written - self.size

            out = np.zeros(avail, dtype=np.float32)
            start_idx = read_index % self.size
            if start_idx + avail <= self.size:
                out[:] = self.buffer[start_idx:start_idx + avail]
            else:
                p1 = self.size - start_idx
                out[:p1] = self.buffer[start_idx:]
                out[p1:] = self.buffer[:avail - p1]
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
                p1 = self.size - start_idx
                out[:p1] = self.buffer[start_idx:]
                out[p1:] = self.buffer[:self.write_index]
            return out

# =====================================================================
# 4. AUDIO FEATURE EXTRACTION (Log-Mel Spectrogram)
# =====================================================================

def compute_log_mel_spectrogram(audio_16k, target_frames=251):
    # Normalized audio
    max_amp = np.max(np.abs(audio_16k))
    if max_amp > 0.005:
        audio_16k = audio_16k / max_amp

    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio_16k, sr=16000, n_fft=512, hop_length=160,
            win_length=512, center=True, pad_mode='reflect',
            n_mels=64, htk=True, norm='slaney'
        )
        log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    except ImportError:
        # Fallback basic spectrogram
        from scipy import signal
        f, t, Sxx = signal.spectrogram(audio_16k, fs=16000, nperseg=512, noverlap=352)
        log_mel = np.log10(np.maximum(Sxx[:64, :], 1e-6))

    if log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]
    elif log_mel.shape[1] < target_frames:
        pad = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')

    return log_mel.reshape(1, 1, 64, target_frames).astype(np.float32)

# =====================================================================
# 5. ONLINE OPEN-SOURCE DATA INTEGRATION (Weather)
# =====================================================================

def fetch_live_weather():
    """Fetches real-time weather using Open-Meteo's free open-source API (no key needed)."""
    try:
        # Default coordinates: Manila, Philippines (lat: 14.5995, lon: 120.9842)
        url = "https://api.open-meteo.com/v1/forecast?latitude=14.5995&longitude=120.9842&current=temperature_2m,weather_code,relative_humidity_2m"
        req = urllib.request.Request(url, headers={"User-Agent": "VoiceControlAssistant/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            current = data.get("current", {})
            temp = current.get("temperature_2m", "--")
            humidity = current.get("relative_humidity_2m", "--")
            return f"{temp}°C, Humidity: {humidity}%"
    except Exception as e:
        return f"Weather unavailable offline ({e})"

# =====================================================================
# 6. MAIN VOICE CONTROLLER APPLICATION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Voice Control Assistant")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--mic-id", type=int, default=None, help="Input microphone device ID.")
    parser.add_argument("--mock-gpio", action="store_true", help="Force Mock GPIO mode.")
    parser.add_argument("--wakeword-model", type=str, default="betelgeuse.onnx", help="Wake word model path.")
    parser.add_argument("--vcm-model", type=str, default="vcm_int8.onnx", help="Voice Control ONNX model path.")
    args = parser.parse_args()

    if args.list_devices:
        print("[*] Available audio devices:")
        print(sd.query_devices())
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Audio cues
    sound_start_path = os.path.join(script_dir, "voice_start.ogg")
    sound_success_path = os.path.join(script_dir, "voice_success.ogg")
    sound_unknown_path = os.path.join(script_dir, "voice_unknown.ogg")

    print("[*] Initializing PyGame mixer...")
    pygame.mixer.init()
    cue_channel = pygame.mixer.Channel(1)

    cues = {
        "start": pygame.mixer.Sound(sound_start_path) if os.path.exists(sound_start_path) else None,
        "success": pygame.mixer.Sound(sound_success_path) if os.path.exists(sound_success_path) else None,
        "unknown": pygame.mixer.Sound(sound_unknown_path) if os.path.exists(sound_unknown_path) else None,
    }

    # Music setup
    music_files = [os.path.join(script_dir, f) for f in ["music1.mp3", "music2.mp3"] if os.path.exists(os.path.join(script_dir, f))]
    current_music_idx = 0
    music_playing = False
    music_volume = 0.8

    # LEDs
    leds = LEDController(force_mock=args.mock_gpio)

    # Load label mappings
    labels_file = os.path.join(script_dir, "intents.json")
    slots_file = os.path.join(script_dir, "slots.json")
    with open(labels_file, 'r', encoding='utf-8') as f:
        intents_list = json.load(f)
    id2intent = {i: name for i, name in enumerate(intents_list)}

    with open(slots_file, 'r', encoding='utf-8') as f:
        slots_data = json.load(f)
    slot_values = ["No Slot"]
    for _, s_info in slots_data.items():
        for val in s_info.get("values", []):
            if val not in slot_values:
                slot_values.append(val)
    id2slot = {i: val for i, val in enumerate(slot_values)}

    # Load ONNX sessions
    vcm_path = os.path.join(script_dir, args.vcm_model)
    if not os.path.exists(vcm_path):
        vcm_path = os.path.join(script_dir, "vcm.onnx")
    print(f"[*] Loading Voice Control ONNX Model from {vcm_path}...")
    vcm_session = ort.InferenceSession(vcm_path, providers=['CPUExecutionProvider'])

    # Wakeword Model
    ww_path = os.path.join(script_dir, args.wakeword_model)
    oww_model = None
    if os.path.exists(ww_path):
        try:
            import openwakeword
            from openwakeword.model import Model
            oww_model = Model(wakeword_models=[ww_path], inference_framework='onnx')
            print(f"[+] Loaded custom wake word model: {ww_path}")
        except Exception as e:
            print(f"[!] Warning: OpenWakeWord could not load ({e}).")
    else:
        try:
            import openwakeword
            from openwakeword.model import Model
            oww_model = Model(wakeword_models=["alexa", "hey_jarvis"], inference_framework='tflite')
            print(f"[*] Custom wake word '{args.wakeword_model}' not found in deployment/. Loaded built-in fallback models ('alexa', 'hey_jarvis').")
        except Exception as e:
            print(f"[!] Warning: OpenWakeWord fallback models could not load ({e}).")

    # Audio stream configuration
    mic_device = args.mic_id
    if mic_device is None:
        try:
            default_dev = sd.query_devices(kind='input')
            mic_device = default_dev['index'] if 'index' in default_dev else 0
        except Exception:
            mic_device = 0

    dev_info = sd.query_devices(mic_device, 'input')
    mic_sr = int(dev_info.get('default_samplerate', 44100))
    print(f"[*] Selected Mic ID {mic_device}: {dev_info.get('name')} (Sample Rate: {mic_sr} Hz)")

    buffer_size = 16000 * 4  # 4 seconds rolling buffer
    ring_buffer = AudioRingBuffer(buffer_size)
    audio_broadcaster = AudioBroadcaster()

    def audio_callback(indata, frames, time_info, status):
        audio_in = indata[:, 0]
        if mic_sr != 16000:
            import scipy.signal
            num_target = int(len(audio_in) * 16000 / mic_sr)
            audio_16k = scipy.signal.resample(audio_in, num_target)
        else:
            audio_16k = audio_in

        # Audio level computation for abstract circle indicator
        rms = float(np.sqrt(np.mean(audio_16k ** 2)))
        norm_level = float(np.clip((rms - 0.002) * 8.5, 0.0, 1.0))
        audio_broadcaster.update_level(norm_level)

        ring_buffer.write(audio_16k.astype(np.float32))

    block_size = int(mic_sr * 0.08)
    stream = sd.InputStream(
        device=mic_device,
        samplerate=mic_sr,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=block_size
    )
    stream.start()

    print("\n" + "=" * 65)
    print(" Voice Controller Running - Say 'Betelgeuse' to begin! ")
    print("=" * 65)

    STATE_LISTENING = 0
    STATE_RECORDING = 1
    STATE_INFERENCE = 2

    state = STATE_LISTENING
    read_index = 0
    recording_start_sample = 0
    was_playing_before_wake = False
    command_samples_target = int(16000 * 2.5)

    send_web_event_async("api/status", {"state": "LISTENING", "details": "LISTENING FOR \"BETELGEUSE\""})

    try:
        while True:
            current_time = time.time()

            if state == STATE_LISTENING:
                recent_audio, read_index = ring_buffer.read_since(read_index)
                if oww_model and len(recent_audio) > 0:
                    audio_int16 = np.clip(recent_audio * 32767 * 2.0, -32768, 32767).astype(np.int16)
                    pred = oww_model.predict(audio_int16)
                    conf = max(pred.values()) if pred else 0.0

                    if conf > 0.04:
                        print(f"\n[!] WAKE WORD DETECTED ('Betelgeuse', Conf: {conf:.2f})")
                        audio_broadcaster.set_state("RECORDING")
                        send_web_event_async("api/status", {"state": "RECORDING", "details": "LISTENING TO COMMAND..."})

                        # Auto-pause music
                        was_playing_before_wake = False
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.pause()
                            was_playing_before_wake = True
                            music_playing = False

                        # Play start audio cue (non-interfering)
                        if cues["start"]:
                            cue_channel.play(cues["start"])
                            # Wait for sound to finish before capturing command audio so it doesn't bleed into mic
                            while cue_channel.get_busy():
                                time.sleep(0.05)

                        # Set recording start point AFTER chime finishes
                        with ring_buffer.lock:
                            recording_start_sample = ring_buffer.total_written
                        state = STATE_RECORDING

            elif state == STATE_RECORDING:
                with ring_buffer.lock:
                    samples_captured = ring_buffer.total_written - recording_start_sample

                if samples_captured >= command_samples_target:
                    audio_broadcaster.set_state("INFERENCE")
                    send_web_event_async("api/status", {"state": "INFERENCE", "details": "ANALYZING COMMAND..."})
                    state = STATE_INFERENCE

            elif state == STATE_INFERENCE:
                t0 = time.perf_counter()
                cmd_audio = ring_buffer.read_latest(command_samples_target)
                input_tensor = compute_log_mel_spectrogram(cmd_audio)

                input_name = vcm_session.get_inputs()[0].name
                outputs = vcm_session.run(None, {input_name: input_tensor})
                intent_logits = outputs[0]
                slot_logits = outputs[1]

                t1 = time.perf_counter()
                infer_latency = (t1 - t0) * 1000.0

                intent_id = int(np.argmax(intent_logits))
                slot_id = int(np.argmax(slot_logits))

                # Softmax confidence
                exp_intent = np.exp(intent_logits - np.max(intent_logits))
                intent_conf = float(np.max(exp_intent) / np.sum(exp_intent) * 100.0)

                detected_intent = id2intent.get(intent_id, "UNKNOWN")
                detected_slot = id2slot.get(slot_id, "No Slot")

                if intent_conf < 30.0:
                    detected_intent = "UNKNOWN"

                print(f"[+] Recognized: Intent={detected_intent} ({intent_conf:.1f}%) | Slot={detected_slot} | Latency={infer_latency:.1f}ms")

                # Audio cues and Web synchronization
                if detected_intent == "UNKNOWN":
                    if cues["unknown"]:
                        cue_channel.play(cues["unknown"])
                else:
                    if cues["success"]:
                        cue_channel.play(cues["success"])

                send_web_event_async("api/command", {
                    "intent": detected_intent,
                    "slot": detected_slot if detected_slot != "No Slot" else "",
                    "confidence": intent_conf,
                    "latency_ms": infer_latency
                })

                # =====================================================
                # EXECUTE INTENT ACTIONS
                # =====================================================
                if detected_intent == "LIGHT_ON":
                    leds.on_all()
                elif detected_intent == "LIGHT_OFF":
                    leds.off_all()
                elif detected_intent == "COLOR":
                    leds.set_color(detected_slot)
                elif detected_intent == "BRIGHTNESS":
                    try:
                        pct = float(detected_slot.replace("%", "").split()[0]) / 100.0
                        leds.set_brightness(pct)
                    except Exception:
                        pass
                elif detected_intent == "PLAY_MUSIC":
                    if music_files:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.load(music_files[current_music_idx])
                            pygame.mixer.music.set_volume(music_volume)
                            pygame.mixer.music.play()
                        else:
                            pygame.mixer.music.unpause()
                        music_playing = True
                        send_web_event_async("api/music", {"status": "playing", "track": os.path.basename(music_files[current_music_idx]), "volume": music_volume})
                elif detected_intent == "PAUSE":
                    pygame.mixer.music.pause()
                    music_playing = False
                    send_web_event_async("api/music", {"status": "paused", "volume": music_volume})
                elif detected_intent == "STOP":
                    pygame.mixer.music.stop()
                    music_playing = False
                    send_web_event_async("api/music", {"status": "stopped", "volume": music_volume})
                elif detected_intent == "NEXT":
                    if music_files:
                        current_music_idx = (current_music_idx + 1) % len(music_files)
                        pygame.mixer.music.load(music_files[current_music_idx])
                        pygame.mixer.music.play()
                        music_playing = True
                        send_web_event_async("api/music", {"status": "playing", "track": os.path.basename(music_files[current_music_idx]), "volume": music_volume})
                elif detected_intent == "VOLUME_UP":
                    music_volume = min(1.0, music_volume + 0.2)
                    pygame.mixer.music.set_volume(music_volume)
                    send_web_event_async("api/music", {"volume": music_volume})
                elif detected_intent == "VOLUME_DOWN":
                    music_volume = max(0.0, music_volume - 0.2)
                    pygame.mixer.music.set_volume(music_volume)
                    send_web_event_async("api/music", {"volume": music_volume})
                elif detected_intent == "WEATHER":
                    weather_info = fetch_live_weather()
                    print(f"  [Live Weather]: {weather_info}")
                    send_web_event_async("api/context", {"text": f"Weather: {weather_info}"})
                elif detected_intent == "TIME":
                    time_str = time.strftime("%I:%M:%S %p")
                    print(f"  [Current Time]: {time_str}")
                    send_web_event_async("api/context", {"text": f"Time: {time_str}"})

                # Auto-resume music if was playing before wake
                if was_playing_before_wake and detected_intent not in ["PAUSE", "STOP"]:
                    pygame.mixer.music.unpause()
                    music_playing = True
                    send_web_event_async("api/music", {"status": "playing", "volume": music_volume})

                if oww_model:
                    oww_model.reset()

                with ring_buffer.lock:
                    read_index = ring_buffer.total_written
                state = STATE_LISTENING
                audio_broadcaster.set_state("LISTENING")
                send_web_event_async("api/status", {"state": "LISTENING", "details": "LISTENING FOR \"BETELGEUSE\""})

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        audio_broadcaster.stop()
        stream.stop()
        stream.close()
        leds.close()
        pygame.mixer.quit()
        print("[+] Hardware cleaned up.")

if __name__ == "__main__":
    main()

