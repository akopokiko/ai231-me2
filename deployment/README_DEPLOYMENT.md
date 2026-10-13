# Voice Control Assistant - Deployment & Setup Guide

This deployment package contains the complete offline Voice Control Assistant and real-time Web Dashboard, ready to run on both a **Windows Laptop / PC** and a **Raspberry Pi 5**.

---

## 1. Package Contents

- `voice_controller.py`: Unified voice assistant controller with automatic Hardware Abstraction Layer (HAL). Auto-mocks GPIO on PC/laptop and controls physical GPIO on Raspberry Pi 5.
- `web_server.py`: Real-time Flask dashboard with Server-Sent Events (SSE).
- `templates/index.html`: Web dashboard displaying status, commands, virtual RGB LEDs with PWM glow, music player, and activity logs.
- `voice_start.ogg`: Audio cue played when wake word "Betelgeuse" is recognized (audio buffer is advanced past this chime to prevent echo).
- `voice_success.ogg`: Audio cue played when a command is recognized.
- `voice_unknown.ogg`: Audio cue played when a command is unrecognized.
- `betelgeuse.onnx`: Custom openwakeword wake word model.
- `vcm_int8.onnx` / `vcm.onnx`: INT8 quantized and FP32 multi-head BC-ResNet models for intent and slot classification.
- `intents.json` & `slots.json`: Synchronized label and slot schemas.

---

## 2. Quickstart on Windows Laptop / PC

### Step 2.1: Environment Setup
Using Conda:
```bash
conda env create -f environment_pc.yml
conda activate vcm_pc
```
Or using standard virtual environment:
```bash
python -m venv vcm_env
vcm_env\Scripts\activate
pip install -r requirements_pc.txt
```

### Step 2.2: Launch Web Dashboard
In terminal 1:
```bash
python web_server.py
```
Open your browser at `http://127.0.0.1:5000`.

### Step 2.3: Launch Voice Controller
In terminal 2:
```bash
# Check microphone devices
python voice_controller.py --list-devices

# Launch with default or chosen mic ID
python voice_controller.py --mock-gpio
```

### Step 2.4: Testing on Laptop
1. Say **"Betelgeuse"** -> Hear start chime, dashboard status changes to `RECORDING`.
2. Say **"Turn on the lights"** -> Hear success chime, virtual LEDs light up in the browser!
3. Say **"Change color to red"** -> Virtual Red LED glows at 100%.
4. Say **"Play music"** -> PyGame plays music, dashboard shows track title.
5. Say **"What's the weather?"** -> Free Open-Meteo API query retrieves real-time temperature and conditions.

---

## 3. Quickstart on Raspberry Pi 5

### Step 3.1: System Dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev libasound2-dev git
```

### Step 3.2: Python Environment Setup
```bash
python3 -m venv ~/vcm_pi
source ~/vcm_pi/bin/activate
pip install -r requirements_rpi.txt
```

### Step 3.3: Hardware Wiring (LEDs on GPIO)
Connect 3 LEDs (Red, Green, Blue) with 220Ω current-limiting resistors:

| LED Color | Positive (Anode) via 220Ω Resistor | BCM GPIO Pin | Physical Board Pin | Negative (Cathode) |
| :--- | :--- | :--- | :--- | :--- |
| **RED** | Resistor to Anode | **GPIO 17** | **Pin 11** | GND (Pin 9 or 14) |
| **GREEN** | Resistor to Anode | **GPIO 27** | **Pin 13** | GND (Pin 9 or 14) |
| **BLUE** | Resistor to Anode | **GPIO 22** | **Pin 15** | GND (Pin 9 or 14) |

### Step 3.4: Bluetooth Speaker Pairing
Pair your Bluetooth speaker using `bluetoothctl`:
```bash
bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
[bluetooth]# scan on
# (Put your Bluetooth speaker into pairing mode and note its MAC address, e.g. 11:22:33:44:55:66)
[bluetooth]# pair 11:22:33:44:55:66
[bluetooth]# trust 11:22:33:44:55:66
[bluetooth]# connect 11:22:33:44:55:66
[bluetooth]# exit
```

### Step 3.5: USB Microphone Setup
Plug in your USB microphone and list audio devices:
```bash
python voice_controller.py --list-devices
```
Find the index of your USB microphone (e.g., `device=1`), then start the controller:
```bash
python voice_controller.py --mic-id 1
```

### Step 3.6: Running as Background Services
You can run both `web_server.py` and `voice_controller.py` as systemd services or with `tmux`/`screen` so they start automatically on boot.

