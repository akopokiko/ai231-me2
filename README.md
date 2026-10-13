# Voice Control Model (BC-ResNet) // Betelgeuse

An end-to-end, privacy-preserving, offline voice control assistant with joint intent classification and slot filling, custom wake word detection ("Betelgeuse"), and real-time smart dashboard integration.

Trained on a university DGX server (8x NVIDIA A100 GPUs), tested on Windows laptops, and deployable to a Raspberry Pi 5 with hardware GPIO LEDs, Bluetooth speaker, and USB microphone.

---

## 1. System Overview

```
                      ┌───────────────────────────────────────┐
                      │    Microphone Input (16 kHz Mono)     │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │    Stage 1: OpenWakeWord Detector     │
                      │         Trigger: "Betelgeuse"         │
                      └──────────────────┬────────────────────┘
                                         ▼  (Plays voice_start.ogg & pauses music)
                      ┌───────────────────────────────────────┐
                      │     Stage 2: 2.5s Audio Capture       │
                      │         Log-Mel Spectrogram           │
                      │            (1 x 64 x 251)             │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   Multi-Head BC-ResNet (ONNX INT8)    │
                      ├──────────────────┬────────────────────┤
                      │  20 Intents Head │   19 Slots Head    │
                      └─────────┬────────┴──────────┬─────────┘
                                │                   │
                                ▼                   ▼
                   ┌─────────────────────────────────────────┐
                   │       Hardware & UI Dispatch (HAL)      │
                   ├─────────────────────────────────────────┤
                   │ • GPIO LEDs: Red (17), Green (27), Blue(22)
                   │ • Music Player: PyGame (play/pause/skip) │
                   │ • Online Data: Real-Time Open-Meteo API │
                   │ • Web Dashboard: Live SSE at :5000       │
                   └─────────────────────────────────────────┘
```

---

## 2. Repository Layout

```
.
├── README.md                      # This guide
├── model.py                       # Multi-Head BC-ResNet architecture
├── dataset.py                     # Dataset loader with online SpecAugment & MUSAN noise
├── preprocess.py                  # 16 kHz audio resampler
├── generate_unknowns.py           # Hybrid UNKNOWN generator (speech + noise + silence)
├── record_laptop_voice.py         # Portable laptop voice recording utility
├── package_finetune_data.py       # Helper to package finetune_16k/ recordings to ZIP
├── train.py                       # DGX A100 training script (auto/manual GPU selection)
├── eval_and_export.py             # ONNX FP32/INT8 export & test set benchmarking
├── scripts/
│   ├── download_baseline_data.py  # Sparse checkout downloader for MEX2 dataset
│   ├── download_musan.py          # MUSAN dataset downloader & extractor
│   └── prepare_dataset.py         # Unified dataset assembler & multi-speaker ingestion
├── data/
│   ├── voice_dataset/             # Preprocessed 16kHz baseline audio files
│   ├── musan/                     # MUSAN noise, speech, and music folders
│   ├── custom_voices/             # User and friends' recorded audio folders
│   ├── manifest_unified.csv       # Unified dataset manifest (18,244+ samples)
│   ├── intents.json               # Canonical 20 intents mapping
│   └── slots.json                 # Canonical 19 slot values mapping
├── metrics/                       # Checkpoints, training curves, and benchmark JSONs
├── wakeword/                      # Custom wake word training pipeline (OpenWakeWord)
└── deployment/                    # Self-contained bundle for Laptop and Raspberry Pi 5
    ├── voice_controller.py        # Edge controller with Hardware Abstraction Layer
    ├── web_server.py              # Flask real-time dashboard server (SSE)
    ├── templates/index.html       # Web dashboard frontend
    ├── voice_start.ogg            # Wake word chime (non-interfering)
    ├── voice_success.ogg          # Valid intent chime
    ├── voice_unknown.ogg          # Unknown/rejected intent chime
    ├── betelgeuse.onnx            # Wake word model
    ├── vcm_int8.onnx              # Quantized INT8 BC-ResNet model
    ├── environment_pc.yml         # Conda setup for Windows/PC
    ├── requirements_pc.txt        # Pip requirements for PC
    ├── requirements_rpi.txt       # Pip requirements for Raspberry Pi 5
    └── README_DEPLOYMENT.md       # Hardware wiring & edge deployment guide
```

---

## 3. Environment Setup (DGX Server)

The training pipeline uses an isolated Conda environment named **`env_me2`**:

```bash
# Check available environments
conda env list

# Activate environment
conda activate env_me2
# Or directly invoke via the environment's python binary:
# /home/francis.neil.jarical/.conda/envs/env_me2/bin/python
```

Verify GPU availability across the 8x A100 GPUs:
```bash
python train.py --check_gpu
```

---

## 4. How to Record & Add Custom Voice Data

You can record your own voice and also record friends' voices to improve acoustic generalization across pitches, accents, and speaking styles.

### Step 4.1: Record on Your Laptop
Download or open [`record_laptop_voice.py`](file:///home/francis.neil.jarical/me2/record_laptop_voice.py) on your laptop (Windows / Mac / Linux):
```bash
pip install sounddevice numpy
python record_laptop_voice.py
```

1. **Enter a Speaker ID**:
   - For yourself: `s_kiko` (or `s_user`).
   - For a friend: `s_friend1`, `s_alice`, etc.
2. Select **`1`** (`Record all phrases sequentially`):
   - The script prompts you phrase by phrase (e.g., *"Turn on the lights"*, *"Change color to red"*).
   - Audio is saved directly at 16 kHz 16-bit mono.
3. When finished, select **`3`** (`Package recordings to ZIP archive`):
   - Creates a portable zip: `custom_voice_<speaker_id>.zip` (e.g., `custom_voice_s_kiko.zip`).

*(Note: If you already recorded audio into `finetune_16k/`, simply run `python package_finetune_data.py` to package those files directly).*

### Step 4.2: Upload to the DGX Server
Transfer the `.zip` archive to the DGX server and unpack it inside `data/custom_voices/`:
```bash
unzip data/custom_voices/custom_voice_s_kiko.zip -d data/custom_voices/
```
The directory structure will be:
```
data/custom_voices/
└── s_kiko/
    ├── manifest_16k.csv (or manifest.csv)
    ├── LIGHT_ON_16k/
    ├── COLOR_red_16k/
    └── ...
```

### Step 4.3: Assemble the Unified Dataset
Run the ingestion script:
```bash
python scripts/prepare_dataset.py
```
This automatically:
- Scans `data/custom_voices/` for all custom speaker folders.
- Stratifies your audio with 80% train, 10% val, 10% test splits.
- Blends your recordings with baseline audio and hybrid UNKNOWN samples into `data/manifest_unified.csv`.
- Synchronizes `intents.json` and `slots.json` into both `data/` and `deployment/`.

---

## 5. Training the Model on DGX

### 5.1: Automatic or Manual GPU Selection
The DGX server does not use SLURM. You have two options:
- **Automatic (Recommended)**: The script queries `nvidia-smi` and picks the GPU with the most free memory.
- **Manual**: Specify the exact GPU index using `--gpu <0-7>`.

```bash
# Auto-select GPU:
python train.py --epochs 25 --batch_size 64

# Or target a specific GPU (e.g., GPU 3):
python train.py --gpu 3 --epochs 25 --batch_size 64
```

### 5.2: Augmentation for Custom Voices
During training, **all custom voice recordings are dynamically augmented on every epoch**:
- **MUSAN Noise Injection**: Randomly mixes background noise/speech from MUSAN at a randomized SNR (5 dB to 15 dB).
- **Time Shifting**: Randomly shifts waveforms left/right by up to ±0.25 seconds.
- **SpecAugment**: Applies random frequency and time masking on the Log-Mel spectrogram.

### 5.3: Incremental Retraining (Adding New Friends Later)
When a friend records their voice later, you don't have to train from scratch for 25 epochs:
```bash
# Fine-tune existing checkpoint on the updated dataset in just 5–10 epochs:
python train.py --resume_from metrics/best_model.pt --epochs 10 --lr 1e-4
```

### 5.4: Outputs
Training automatically generates:
- `metrics/best_model.pt`: Checkpoint with highest validation sentence exact match accuracy.
- `metrics/training_curves.png`: Training & validation loss and accuracy curves.
- `metrics/training_history.csv`: Per-epoch metrics table.
- `metrics/training_summary.json`: High-level performance summary.

---

## 6. ONNX Export & Test Benchmarking

Export the PyTorch checkpoint to ONNX and evaluate accuracy across the test split:
```bash
python eval_and_export.py
```

This generates:
1. `deployment/vcm.onnx`: Full-precision FP32 ONNX model.
2. `deployment/vcm_int8.onnx`: Dynamic INT8 quantized ONNX model (optimized for Raspberry Pi 5 CPU).
3. `metrics/confusion_matrix_intent.png` & `metrics/confusion_matrix_slot.png`.
4. `metrics/test_metrics.json`: Accuracy scores and average CPU latency.

---

## 7. Testing Locally on Laptop (Windows / Mac / PC)

Download the [`deployment/`](file:///home/francis.neil.jarical/me2/deployment) folder to your laptop.

### 7.1: Start the Web Dashboard
```bash
python deployment/web_server.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### 7.2: Start the Voice Controller (Mock GPIO Mode)
```bash
python deployment/voice_controller.py --mock-gpio
```
- Say **"Betelgeuse"** -> Start chime plays, UI switches to `RECORDING`.
- Say **"Turn on the lights"** -> Virtual RGB LEDs illuminate in the web dashboard.
- Say **"Change color to green"** -> Green LED glows at 100%.
- Say **"What's the weather?"** -> Free Open-Meteo API query retrieves live temperature and broadcasts it to the dashboard.
- Say **"Play music"** -> PyGame plays music, track title appears in the UI.

---

## 8. Edge Deployment on Raspberry Pi 5

Follow the complete hardware wiring, Bluetooth pairing, and service setup in [`deployment/README_DEPLOYMENT.md`](file:///home/francis.neil.jarical/me2/deployment/README_DEPLOYMENT.md).

### Quick Commands on Raspberry Pi 5:
```bash
# 1. Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv portaudio19-dev libasound2-dev
python3 -m venv ~/vcm_pi
source ~/vcm_pi/bin/activate
pip install -r deployment/requirements_rpi.txt

# 2. Pair Bluetooth Speaker
bluetoothctl
[bluetooth]# scan on
[bluetooth]# pair <MAC_ADDRESS>
[bluetooth]# connect <MAC_ADDRESS>

# 3. Check USB microphone device index
python deployment/voice_controller.py --list-devices

# 4. Start Dashboard and Voice Controller
python deployment/web_server.py &
python deployment/voice_controller.py --mic-id 1
```

Physical GPIO pins used on Raspberry Pi 5:
- **Red LED**: GPIO 17 (Pin 11)
- **Green LED**: GPIO 27 (Pin 13)
- **Blue LED**: GPIO 22 (Pin 15)
- **Ground**: Pin 9 or 14
