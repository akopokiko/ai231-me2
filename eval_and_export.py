#!/usr/bin/env python3
"""
Evaluates trained Voice Control Model and exports to ONNX (FP32 & INT8 quantized).
Computes:
- Top-1 Intent Accuracy
- Slot Accuracy
- Sentence Exact Match Accuracy
- CPU Inference Latency benchmark (FP32 vs INT8)
- Confusion Matrices (saved to metrics/)
- Exports deployment/vcm.onnx and deployment/vcm_int8.onnx
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import BCResNet
from dataset import ChatterboxDataset

def export_onnx(model, dummy_input, output_path="deployment/vcm.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    print(f"[*] Exporting PyTorch model to FP32 ONNX: {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['intent_logits', 'slot_logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'intent_logits': {0: 'batch_size'},
            'slot_logits': {0: 'batch_size'}
        }
    )

    # Check onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] FP32 ONNX exported successfully ({size_mb:.2f} MB).")
    return output_path

def quantize_to_int8(input_path="deployment/vcm.onnx", output_path="deployment/vcm_int8.onnx"):
    print(f"[*] Quantizing to INT8 dynamic ONNX: {output_path}...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] INT8 ONNX exported successfully ({size_mb:.2f} MB).")
    return output_path

def evaluate_onnx_session(session, test_loader, num_intents, num_slots):
    all_intent_preds = []
    all_intent_targets = []
    all_slot_preds = []
    all_slot_targets = []
    latencies = []

    input_name = session.get_inputs()[0].name

    for mels, intent_targets, slot_targets in test_loader:
        mels_np = mels.numpy()

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: mels_np})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0 / len(mels_np))

        intent_logits, slot_logits = outputs[0], outputs[1]
        pred_intents = np.argmax(intent_logits, axis=1)
        pred_slots = np.argmax(slot_logits, axis=1)

        all_intent_preds.extend(pred_intents.tolist())
        all_intent_targets.extend(intent_targets.numpy().tolist())
        all_slot_preds.extend(pred_slots.tolist())
        all_slot_targets.extend(slot_targets.numpy().tolist())

    all_intent_preds = np.array(all_intent_preds)
    all_intent_targets = np.array(all_intent_targets)
    all_slot_preds = np.array(all_slot_preds)
    all_slot_targets = np.array(all_slot_targets)

    intent_match = (all_intent_preds == all_intent_targets)
    slot_match = (all_slot_preds == all_slot_targets)
    exact_match = intent_match & slot_match

    return {
        "intent_accuracy": float(np.mean(intent_match) * 100.0),
        "slot_accuracy": float(np.mean(slot_match) * 100.0),
        "exact_match_accuracy": float(np.mean(exact_match) * 100.0),
        "avg_latency_ms": float(np.mean(latencies)),
        "all_intent_preds": all_intent_preds,
        "all_intent_targets": all_intent_targets,
        "all_slot_preds": all_slot_preds,
        "all_slot_targets": all_slot_targets
    }

def plot_confusion_matrix(targets, preds, labels, title, output_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds, labels=range(len(labels)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Voice Control Model and export ONNX")
    parser.add_argument("--checkpoint", type=str, default="metrics/best_model.pt", help="Path to trained PyTorch checkpoint")
    parser.add_argument("--manifest", type=str, default="data/manifest_unified.csv", help="Test manifest CSV")
    parser.add_argument("--labels", type=str, default="data/intents.json", help="Intents mapping JSON")
    parser.add_argument("--slots", type=str, default="data/slots.json", help="Slots mapping JSON")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[-] Checkpoint not found: {args.checkpoint}. Run train.py first.")
        sys.exit(1)

    with open(args.labels, 'r', encoding='utf-8') as f:
        intents_list = json.load(f)
    num_intents = len(intents_list)

    with open(args.slots, 'r', encoding='utf-8') as f:
        slots_data = json.load(f)
    slot_values = ["No Slot"]
    for _, s_info in slots_data.items():
        for val in s_info.get("values", []):
            if val not in slot_values:
                slot_values.append(val)
    num_slots = len(slot_values)

    print(f"[*] Loading checkpoint: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = BCResNet(num_intents=num_intents, num_slots=num_slots)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Export FP32 ONNX
    dummy_input = torch.randn(1, 1, 64, 251)
    fp32_path = export_onnx(model, dummy_input, "deployment/vcm.onnx")

    # 2. Export INT8 ONNX
    int8_path = quantize_to_int8(fp32_path, "deployment/vcm_int8.onnx")

    # 3. Benchmark on Test Set
    print(f"\n[*] Evaluating on Test Set...")
    test_dataset = ChatterboxDataset(
        manifest_path=args.manifest,
        labels_path=args.labels,
        slots_path=args.slots,
        base_dir=".",
        split="test",
        augment=False,
        device="cpu"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"    Test set size: {len(test_dataset)} samples.")

    # Evaluate FP32 ONNX
    session_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    results_fp32 = evaluate_onnx_session(session_fp32, test_loader, num_intents, num_slots)

    # Evaluate INT8 ONNX
    session_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
    results_int8 = evaluate_onnx_session(session_int8, test_loader, num_intents, num_slots)

    print("\n" + "=" * 65)
    print(" Test Set Evaluation Results ")
    print("=" * 65)
    print(f"Model: FP32 ONNX ({fp32_path})")
    print(f"  - Intent Top-1 Accuracy       : {results_fp32['intent_accuracy']:.2f}%")
    print(f"  - Slot Accuracy               : {results_fp32['slot_accuracy']:.2f}%")
    print(f"  - Sentence Exact Match Acc    : {results_fp32['exact_match_accuracy']:.2f}%")
    print(f"  - Average CPU Latency         : {results_fp32['avg_latency_ms']:.2f} ms")

    print(f"\nModel: INT8 ONNX ({int8_path})")
    print(f"  - Intent Top-1 Accuracy       : {results_int8['intent_accuracy']:.2f}%")
    print(f"  - Slot Accuracy               : {results_int8['slot_accuracy']:.2f}%")
    print(f"  - Sentence Exact Match Acc    : {results_int8['exact_match_accuracy']:.2f}%")
    print(f"  - Average CPU Latency         : {results_int8['avg_latency_ms']:.2f} ms")
    print("=" * 65)

    # Plot Confusion Matrices for INT8 model
    try:
        plot_confusion_matrix(
            results_int8["all_intent_targets"],
            results_int8["all_intent_preds"],
            intents_list,
            "Intent Confusion Matrix (INT8)",
            "metrics/confusion_matrix_intent.png"
        )
        plot_confusion_matrix(
            results_int8["all_slot_targets"],
            results_int8["all_slot_preds"],
            slot_values,
            "Slot Confusion Matrix (INT8)",
            "metrics/confusion_matrix_slot.png"
        )
    except Exception as e:
        print(f"[!] Warning: Could not generate confusion matrices ({e})")

    # Export metrics JSON
    metrics_summary = {
        "test_samples": len(test_dataset),
        "fp32_onnx": {
            "intent_accuracy": results_fp32["intent_accuracy"],
            "slot_accuracy": results_fp32["slot_accuracy"],
            "exact_match_accuracy": results_fp32["exact_match_accuracy"],
            "avg_latency_ms": results_fp32["avg_latency_ms"],
            "model_size_mb": round(os.path.getsize(fp32_path) / (1024 * 1024), 2)
        },
        "int8_onnx": {
            "intent_accuracy": results_int8["intent_accuracy"],
            "slot_accuracy": results_int8["slot_accuracy"],
            "exact_match_accuracy": results_int8["exact_match_accuracy"],
            "avg_latency_ms": results_int8["avg_latency_ms"],
            "model_size_mb": round(os.path.getsize(int8_path) / (1024 * 1024), 2)
        }
    }

    with open("metrics/test_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved metrics to metrics/test_metrics.json")

if __name__ == "__main__":
    main()

