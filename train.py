#!/usr/bin/env python3
"""
Training script for Multi-Head BC-ResNet Voice Control Model on DGX A100.
Features:
- Automatic or manual GPU selection (--gpu <id>) via nvidia-smi.
- Joint Intent and Slot Classification.
- Metrics tracking: Total Loss, Intent Accuracy, Slot Accuracy, Sentence Exact Match.
- Saves best checkpoint (metrics/best_model.pt), history CSV, and training plots (metrics/training_curves.png).
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BCResNet
from dataset import ChatterboxDataset

def select_gpu(requested_gpu=None):
    if not torch.cuda.is_available():
        print("[!] No CUDA available, using CPU.")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    print(f"[*] Found {num_gpus} CUDA device(s).")

    if requested_gpu is not None:
        if 0 <= requested_gpu < num_gpus:
            print(f"[+] Using requested GPU {requested_gpu}: {torch.cuda.get_device_name(requested_gpu)}")
            return torch.device(f"cuda:{requested_gpu}")
        else:
            print(f"[!] Invalid GPU ID {requested_gpu}. Available: 0 to {num_gpus-1}.")

    # Auto-select GPU with lowest memory usage via nvidia-smi
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd).decode("utf-8").strip().split("\n")
        gpu_mem = []
        for line in output:
            parts = line.split(",")
            idx = int(parts[0].strip())
            used_mb = float(parts[1].strip())
            gpu_mem.append((idx, used_mb))

        gpu_mem.sort(key=lambda x: x[1])
        best_gpu, mem_used = gpu_mem[0]
        print(f"[+] Auto-selected GPU {best_gpu} ({torch.cuda.get_device_name(best_gpu)}) with lowest usage: {mem_used:.0f} MiB")
        return torch.device(f"cuda:{best_gpu}")
    except Exception as e:
        print(f"[!] Could not query nvidia-smi ({e}), falling back to cuda:0.")
        return torch.device("cuda:0")

def train_epoch(model, dataloader, optimizer, criterion_intent, criterion_slot, slot_weight, device):
    model.train()
    total_loss = 0.0
    total_intent_loss = 0.0
    total_slot_loss = 0.0
    correct_intent = 0
    correct_slot = 0
    correct_exact = 0
    total_samples = 0

    for mels, intent_targets, slot_targets in dataloader:
        mels = mels.to(device)
        intent_targets = intent_targets.to(device)
        slot_targets = slot_targets.to(device)

        optimizer.zero_grad()
        intent_logits, slot_logits = model(mels)

        loss_intent = criterion_intent(intent_logits, intent_targets)
        loss_slot = criterion_slot(slot_logits, slot_targets)
        loss = loss_intent + slot_weight * loss_slot

        loss.backward()
        optimizer.step()

        batch_size = mels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_intent_loss += loss_intent.item() * batch_size
        total_slot_loss += loss_slot.item() * batch_size

        pred_intents = intent_logits.argmax(dim=1)
        pred_slots = slot_logits.argmax(dim=1)

        intent_match = (pred_intents == intent_targets)
        slot_match = (pred_slots == slot_targets)
        exact_match = intent_match & slot_match

        correct_intent += intent_match.sum().item()
        correct_slot += slot_match.sum().item()
        correct_exact += exact_match.sum().item()

    return {
        "loss": total_loss / total_samples,
        "loss_intent": total_intent_loss / total_samples,
        "loss_slot": total_slot_loss / total_samples,
        "intent_acc": (correct_intent / total_samples) * 100.0,
        "slot_acc": (correct_slot / total_samples) * 100.0,
        "exact_match": (correct_exact / total_samples) * 100.0
    }

def evaluate(model, dataloader, criterion_intent, criterion_slot, slot_weight, device):
    model.eval()
    total_loss = 0.0
    total_intent_loss = 0.0
    total_slot_loss = 0.0
    correct_intent = 0
    correct_slot = 0
    correct_exact = 0
    total_samples = 0

    with torch.no_grad():
        for mels, intent_targets, slot_targets in dataloader:
            mels = mels.to(device)
            intent_targets = intent_targets.to(device)
            slot_targets = slot_targets.to(device)

            intent_logits, slot_logits = model(mels)

            loss_intent = criterion_intent(intent_logits, intent_targets)
            loss_slot = criterion_slot(slot_logits, slot_targets)
            loss = loss_intent + slot_weight * loss_slot

            batch_size = mels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_intent_loss += loss_intent.item() * batch_size
            total_slot_loss += loss_slot.item() * batch_size

            pred_intents = intent_logits.argmax(dim=1)
            pred_slots = slot_logits.argmax(dim=1)

            intent_match = (pred_intents == intent_targets)
            slot_match = (pred_slots == slot_targets)
            exact_match = intent_match & slot_match

            correct_intent += intent_match.sum().item()
            correct_slot += slot_match.sum().item()
            correct_exact += exact_match.sum().item()

    if total_samples == 0:
        return {"loss": 0, "loss_intent": 0, "loss_slot": 0, "intent_acc": 0, "slot_acc": 0, "exact_match": 0}

    return {
        "loss": total_loss / total_samples,
        "loss_intent": total_intent_loss / total_samples,
        "loss_slot": total_slot_loss / total_samples,
        "intent_acc": (correct_intent / total_samples) * 100.0,
        "slot_acc": (correct_slot / total_samples) * 100.0,
        "exact_match": (correct_exact / total_samples) * 100.0
    }

def plot_curves(history, output_path="metrics/training_curves.png"):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total Loss
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], 'b-o', label="Train Loss")
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], 'r--s', label="Val Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # 2. Sentence Exact Match
    axes[0, 1].plot(epochs, [h["train_exact_match"] for h in history], 'g-o', label="Train Exact Match")
    axes[0, 1].plot(epochs, [h["val_exact_match"] for h in history], 'm--s', label="Val Exact Match")
    axes[0, 1].set_title("Sentence Exact Match Accuracy (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3. Intent Accuracy
    axes[1, 0].plot(epochs, [h["train_intent_acc"] for h in history], 'b-o', label="Train Intent Acc")
    axes[1, 0].plot(epochs, [h["val_intent_acc"] for h in history], 'r--s', label="Val Intent Acc")
    axes[1, 0].set_title("Intent Top-1 Accuracy (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 4. Slot Accuracy
    axes[1, 1].plot(epochs, [h["train_slot_acc"] for h in history], 'g-o', label="Train Slot Acc")
    axes[1, 1].plot(epochs, [h["val_slot_acc"] for h in history], 'm--s', label="Val Slot Acc")
    axes[1, 1].set_title("Slot Accuracy (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Voice Control Model (BC-ResNet)")
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU index (0 to 7). Auto-selects if omitted.")
    parser.add_argument("--check_gpu", action="store_true", help="Print GPU status and exit.")
    parser.add_argument("--manifest", type=str, default="data/manifest_unified.csv", help="Dataset manifest CSV")
    parser.add_argument("--labels", type=str, default="data/intents.json", help="Intents mapping JSON")
    parser.add_argument("--slots", type=str, default="data/slots.json", help="Slots mapping JSON")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--slot_weight", type=float, default=1.0, help="Weight for slot loss")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume or fine-tune from")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    if args.check_gpu:
        select_gpu(args.gpu)
        return

    os.makedirs("metrics", exist_ok=True)
    device = select_gpu(args.gpu)

    # Check manifest
    if not os.path.exists(args.manifest):
        # Fallback to voice_dataset manifest if unified not yet generated
        fallback = "data/voice_dataset/manifest_16k.csv"
        if os.path.exists(fallback):
            args.manifest = fallback
            print(f"[*] Using fallback manifest: {args.manifest}")
        else:
            print(f"[-] Manifest {args.manifest} not found. Please run scripts/prepare_dataset.py first.")
            sys.exit(1)

    with open(args.labels, 'r', encoding='utf-8') as f:
        intents_list = json.load(f)
    num_intents = len(intents_list)

    with open(args.slots, 'r', encoding='utf-8') as f:
        slots_data = json.load(f)
    # Count unique slot values + 1 for "No Slot"
    slot_values = set(["No Slot"])
    for _, s_info in slots_data.items():
        for val in s_info.get("values", []):
            slot_values.add(val)
    num_slots = len(slot_values)

    print(f"[*] Model configuration: num_intents={num_intents}, num_slots={num_slots}")

    # Datasets
    train_dataset = ChatterboxDataset(
        manifest_path=args.manifest,
        labels_path=args.labels,
        slots_path=args.slots,
        base_dir=".",
        split="train",
        augment=True,
        device="cpu"
    )

    val_dataset = ChatterboxDataset(
        manifest_path=args.manifest,
        labels_path=args.labels,
        slots_path=args.slots,
        base_dir=".",
        split="val",
        augment=False,
        device="cpu"
    )

    print(f"[*] Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    # Model
    model = BCResNet(num_intents=num_intents, num_slots=num_slots).to(device)

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[*] Resuming weights from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    criterion_intent = nn.CrossEntropyLoss()
    criterion_slot = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_exact = -1.0
    best_epoch = -1
    history = []

    print("\n" + "=" * 70)
    print(" Beginning Model Training ")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion_intent, criterion_slot, args.slot_weight, device)
        val_metrics = evaluate(model, val_loader, criterion_intent, criterion_slot, args.slot_weight, device)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_intent_acc": round(train_metrics["intent_acc"], 2),
            "train_slot_acc": round(train_metrics["slot_acc"], 2),
            "train_exact_match": round(train_metrics["exact_match"], 2),
            "val_loss": round(val_metrics["loss"], 4),
            "val_intent_acc": round(val_metrics["intent_acc"], 2),
            "val_slot_acc": round(val_metrics["slot_acc"], 2),
            "val_exact_match": round(val_metrics["exact_match"], 2),
            "lr": round(scheduler.get_last_lr()[0], 6)
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_metrics['loss']:.4f} (Int:{train_metrics['intent_acc']:.1f}% Slot:{train_metrics['slot_acc']:.1f}% Exact:{train_metrics['exact_match']:.1f}%) | "
            f"Val Loss: {val_metrics['loss']:.4f} (Int:{val_metrics['intent_acc']:.1f}% Slot:{val_metrics['slot_acc']:.1f}% Exact:{val_metrics['exact_match']:.1f}%)"
        )

        # Checkpointing best model based on validation exact match
        if val_metrics["exact_match"] > best_val_exact:
            best_val_exact = val_metrics["exact_match"]
            best_epoch = epoch
            checkpoint_path = "metrics/best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_exact_match": best_val_exact,
                "val_intent_acc": val_metrics["intent_acc"],
                "val_slot_acc": val_metrics["slot_acc"],
                "num_intents": num_intents,
                "num_slots": num_slots
            }, checkpoint_path)
            print(f"  -> Saved new best model to {checkpoint_path} (Val Exact Match: {best_val_exact:.2f}%)")

    # Save training history
    history_df = pd.DataFrame(history)
    history_csv = "metrics/training_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"\n[+] Saved training history to {history_csv}")

    # Plot curves
    plot_curves(history, "metrics/training_curves.png")

    # Save summary JSON
    summary = {
        "best_epoch": best_epoch,
        "best_val_exact_match": best_val_exact,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_intent_acc": history[-1]["val_intent_acc"],
        "final_val_slot_acc": history[-1]["val_slot_acc"],
        "final_val_exact_match": history[-1]["val_exact_match"]
    }
    with open("metrics/training_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved summary to metrics/training_summary.json")

if __name__ == "__main__":
    main()

