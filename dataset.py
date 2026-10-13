import os
import json
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import soundfile as sf
import numpy as np

class ChatterboxDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, labels_path, slots_path, base_dir, split="train", augment=False, device="cpu"):
        super().__init__()
        self.base_dir = base_dir
        self.augment = augment
        self.device = device
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        self.data = df[df['split'] == split].to_dict('records')
        
        # Load labels mapping
        with open(labels_path, 'r') as f:
            intents = json.load(f)
        self.intent2id = {intent: idx for idx, intent in enumerate(intents)}
        
        # Load slots mapping
        with open(slots_path, 'r') as f:
            slots_data = json.load(f)
            
        # Build slot to id mapping. ID 0 is reserved for "No Slot".
        self.slot2id = {"No Slot": 0}
        slot_idx = 1
        for intent, slot_info in slots_data.items():
            for value in slot_info['values']:
                if value not in self.slot2id:
                    self.slot2id[value] = slot_idx
                    slot_idx += 1
                    
        # Fixed parameters
        self.target_sr = 16000
        self.target_length = 40000  # 2.5 seconds * 16000 Hz
        
        # Feature extraction - moved to specified device
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=512,
            win_length=512,
            hop_length=160,
            n_mels=64,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            mel_scale="htk"
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(self.device)
        
        # Augmentations
        if self.augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15).to(self.device)
            self.time_mask = T.TimeMasking(time_mask_param=35).to(self.device)
            
            # Scan for MUSAN noise files
            self.musan_files = []
            import glob
            for category in ["noise", "speech"]:
                self.musan_files.extend(glob.glob(f"data/musan/{category}/**/*.wav", recursive=True))
                self.musan_files.extend(glob.glob(f"musan/{category}/**/*.wav", recursive=True))

    def _pad_or_crop(self, waveform):
        # waveform shape: (channels, time)
        num_samples = waveform.shape[1]
        
        if num_samples > self.target_length:
            # Center crop
            start = (num_samples - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
        elif num_samples < self.target_length:
            # Pad with ambient background noise
            # Generate soft gaussian noise (ambient-like)
            noise_amp = 0.005 # small amplitude for ambient noise
            
            # Pad equally on both sides
            pad_left = (self.target_length - num_samples) // 2
            pad_right = self.target_length - num_samples - pad_left
            
            left_noise = torch.randn(waveform.shape[0], pad_left, device=waveform.device) * noise_amp
            right_noise = torch.randn(waveform.shape[0], pad_right, device=waveform.device) * noise_amp
            
            waveform = torch.cat([left_noise, waveform, right_noise], dim=1)
            
        return waveform

    def _augment(self, waveform):
        # Time shifting (shift and pad with zeros/silence, without wrapping)
        if random.random() < 0.5:
            shift = random.randint(-4000, 4000)
            if shift > 0:
                # Shift right: pad left with zeros, drop right
                pad_zeros = torch.zeros(waveform.shape[0], shift, device=waveform.device)
                waveform = torch.cat([pad_zeros, waveform[:, :-shift]], dim=1)
            elif shift < 0:
                # Shift left: drop left, pad right with zeros
                pad_zeros = torch.zeros(waveform.shape[0], -shift, device=waveform.device)
                waveform = torch.cat([waveform[:, -shift:], pad_zeros], dim=1)
            
        # Noise injection (MUSAN or Gaussian fallback)
        if random.random() < 0.5:
            if hasattr(self, 'musan_files') and len(self.musan_files) > 0:
                # Pick a random MUSAN file
                noise_file = random.choice(self.musan_files)
                try:
                    noise_np, sr = sf.read(noise_file)
                    if noise_np.ndim > 1:
                        noise_np = noise_np[:, 0] # to mono
                    noise_tensor = torch.from_numpy(noise_np).float().to(waveform.device)
                    
                    # Ensure noise_tensor is (1, time)
                    if noise_tensor.ndim == 1:
                        noise_tensor = noise_tensor.unsqueeze(0)
                        
                    # Match lengths
                    if noise_tensor.shape[1] > waveform.shape[1]:
                        start = random.randint(0, noise_tensor.shape[1] - waveform.shape[1])
                        noise_tensor = noise_tensor[:, start:start+waveform.shape[1]]
                    elif noise_tensor.shape[1] < waveform.shape[1]:
                        # pad noise
                        pad_len = waveform.shape[1] - noise_tensor.shape[1]
                        noise_tensor = torch.nn.functional.pad(noise_tensor, (0, pad_len))
                        
                    # Scale noise to target SNR (e.g., 5 to 15 dB)
                    snr_db = random.uniform(5.0, 15.0)
                    signal_power = waveform.norm(p=2)
                    noise_power = noise_tensor.norm(p=2)
                    
                    if noise_power > 0:
                        scale = signal_power / noise_power * (10 ** (-snr_db / 20.0))
                        waveform = waveform + noise_tensor * scale
                except Exception as e:
                    # Fallback on failure
                    noise = torch.randn_like(waveform) * random.uniform(0.001, 0.015)
                    waveform = waveform + noise
            else:
                noise = torch.randn_like(waveform) * random.uniform(0.001, 0.015)
                waveform = waveform + noise
            
        return waveform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Normalize slashes to fix paths generated on Windows machines
        normalized_path = item['path'].replace('\\', '/')
        audio_path = os.path.join(self.base_dir, normalized_path)
        if not os.path.exists(audio_path) and os.path.exists(normalized_path):
            audio_path = normalized_path
        
        # Bypass torchaudio.load completely to avoid torchcodec incompatibilities
        waveform_np, sr = sf.read(audio_path)
        # Handle cases where waveform is 1D (mono) vs 2D (stereo)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[np.newaxis, :]
        else:
            waveform_np = waveform_np.T # shape: (channels, time)
            
        waveform = torch.from_numpy(waveform_np).float()
        
        # Audio resampling fallback
        if sr != self.target_sr:
            waveform = F.resample(waveform, sr, self.target_sr)
            
        waveform = self._pad_or_crop(waveform)
        
        if self.augment:
            waveform = self._augment(waveform)
            
        # Move waveform to the target device (e.g., GPU) before spectrogram extraction
        waveform = waveform.to(self.device)
            
        # Log-Mel Spectrogram extraction (done on device)
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        if self.augment:
            # SpecAugment (done on device)
            if random.random() < 0.5:
                log_mel_spec = self.freq_mask(log_mel_spec)
            if random.random() < 0.5:
                log_mel_spec = self.time_mask(log_mel_spec)
                
        # Get Intent ID
        intent_id = self.intent2id[item['intent']]
        
        # Get Slot ID
        slot_value = item['slot_value']
        if pd.isna(slot_value) or slot_value == "none" or slot_value == "":
            slot_id = 0
        else:
            slot_id = self.slot2id.get(slot_value, 0)
            
        return log_mel_spec, torch.tensor(intent_id, dtype=torch.long), torch.tensor(slot_id, dtype=torch.long)

if __name__ == "__main__":
    # Simple verification test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ChatterboxDataset(
        manifest_path="manifest_16k.csv",
        labels_path="intents.json",
        slots_path="slots.json",
        base_dir=".",
        split="train",
        augment=True,
        device=device
    )
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        log_mel, intent_id, slot_id = dataset[0]
        print(f"Log-Mel shape: {log_mel.shape} (Expected: (1, 64, 251))")
        print(f"Log-Mel Device: {log_mel.device}")
        print(f"Intent ID: {intent_id}")
        print(f"Slot ID: {slot_id}")
