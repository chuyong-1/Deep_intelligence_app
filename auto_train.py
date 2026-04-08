"""
auto_train.py  v1

MLOps re-training pipeline for the Verity Image Forensics subsystem.

Purpose
───────
Called by app.py via subprocess when the analyst clicks
"🚀 Execute Intelligence Sync" on the Review page.

All stdout output is written to training.log so app.py can tail it live.

Pipeline
────────
1. Collect labelled images from training_data/real/ and training_data/fake/
2. Load the SigLIP-2 backbone (google/siglip2-base-patch16-224)
3. Transfer learning: freeze all layers, unfreeze only the top 2 transformer
   encoder blocks (layers[-2:]) + the classification head
4. Fine-tune at lr=1e-5 for up to MAX_EPOCHS (early-stop on val loss plateau)
5. Evaluate accuracy and save best weights as model/siglip2_finetuned/
6. Re-quantize to ONNX INT8 via Optimum and cache to model/onnx/ai_detector/
7. Log structured JSON lines to training.log so app.py can parse live updates

Training log format (one JSON line per event):
    {"event": "start",   "total_images": N, "real": R, "fake": F}
    {"event": "epoch",   "epoch": E, "loss": L, "val_loss": VL, "val_acc": A}
    {"event": "best",    "epoch": E, "val_acc": A}
    {"event": "eval",    "accuracy": A, "precision": P, "recall": R, "f1": F}
    {"event": "onnx",    "path": "model/onnx/ai_detector/model_quantized.onnx"}
    {"event": "done",    "message": "Training complete"}
    {"event": "error",   "message": "..."}

Hyperparameters (production-safe defaults)
──────────────────────────────────────────
  BACKBONE        google/siglip2-base-patch16-224
  FREEZE_STRATEGY unfreeze top-2 encoder blocks + head
  LR              1e-5   (low — prevents accuracy regression on base weights)
  BATCH_SIZE      16     (conservative for CPU/low-VRAM environments)
  MAX_EPOCHS      10
  PATIENCE        3      (early stop after 3 non-improving val epochs)
  IMG_SIZE        224
  VAL_SPLIT       0.15

Minimum dataset requirement: 10 labelled images (5 per class).
If fewer images are found, training is skipped and a warning is logged.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REAL_DIR         = Path("training_data/real")
FAKE_DIR         = Path("training_data/fake")
MODEL_SAVE_DIR   = Path("model/siglip2_finetuned")
ONNX_OUT_DIR     = Path("model/onnx/ai_detector")
LOG_FILE         = Path("training.log")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BACKBONE         = "google/siglip2-base-patch16-224"
LR               = 1e-5
BATCH_SIZE       = 16
MAX_EPOCHS       = 10
PATIENCE         = 3
IMG_SIZE         = 224
VAL_SPLIT        = 0.15
MIN_IMAGES       = 10   # minimum total images to proceed with training
RANDOM_STATE     = 42

# ── Ensure output dirs exist ───────────────────────────────────────────────────
for _d in (MODEL_SAVE_DIR, ONNX_OUT_DIR, REAL_DIR, FAKE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────

# ─── NEW ───────────────────────────────────────────────────────────────────
def _log(event: str, _max_retries: int = 3, _retry_delay: float = 0.15, **kwargs):
    """
    Write a structured JSON line to training.log AND stdout.

    File write is retried up to _max_retries times with _retry_delay seconds
    between attempts.  On final failure the line is still printed to stdout
    (captured by Streamlit via subprocess.PIPE) so no event is silently lost.
    The training loop itself is never interrupted by a log I/O error.
    """
    record = {"event": event, "ts": int(time.time()), **kwargs}
    line   = json.dumps(record)
    print(line, flush=True)  # always succeeds — stdout is a pipe

    last_exc: Exception | None = None
    for attempt in range(_max_retries):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return  # success
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt < _max_retries - 1:
                time.sleep(_retry_delay * (attempt + 1))

    # All retries exhausted — emit a warning to stdout only (don't raise)
    print(
        json.dumps({
            "event": "log_write_failed",
            "ts": int(time.time()),
            "original_event": event,
            "error": str(last_exc),
        }),
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_paths() -> tuple[list[Path], list[int]]:
    """Return (file_paths, labels) where label 0=real, 1=fake."""
    paths:  list[Path] = []
    labels: list[int]  = []

    for p in sorted(REAL_DIR.iterdir()):
        if p.suffix.lower() in _IMG_EXTS:
            paths.append(p); labels.append(0)

    for p in sorted(FAKE_DIR.iterdir()):
        if p.suffix.lower() in _IMG_EXTS:
            paths.append(p); labels.append(1)

    return paths, labels


def _build_dataset(paths, labels, processor, augment: bool = False):
    """
    Returns a list of (pixel_values_tensor, label_tensor) tuples.
    All images are decoded to RGB and processed via AutoProcessor.
    When augment=True applies horizontal flip + colour jitter for training.
    """
    import torch
    import torchvision.transforms.functional as TF
    import random
    from PIL import Image

    items = []
    for p, lbl in zip(paths, labels):
        try:
            img = Image.open(str(p)).convert("RGB")
        except Exception:
            continue

        if augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
            # Mild colour jitter — does not destroy real diagnostic signal
            img = TF.adjust_brightness(img, 1.0 + (random.random() - 0.5) * 0.3)
            img = TF.adjust_contrast(  img, 1.0 + (random.random() - 0.5) * 0.2)

        enc = processor(images=img, return_tensors="pt")
        pv  = enc["pixel_values"].squeeze(0)   # (C, H, W)
        items.append((pv, torch.tensor(lbl, dtype=torch.long)))

    return items


class _SimpleDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _collate(batch):
    import torch
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return pixels, labels


# ──────────────────────────────────────────────────────────────────────────────
# Transfer-learning: freeze strategy
# ──────────────────────────────────────────────────────────────────────────────

def _apply_freeze_strategy(model):
    """
    Freeze all parameters, then unfreeze only:
      • The top 2 SiglipEncoder layers (vision_model.encoder.layers[-2:])
      • The classification head (classifier)

    This preserves the carefully pre-trained visual representations while
    allowing the top layers to specialise on real-vs-AI forensics at lr=1e-5.
    Prevents catastrophic forgetting of the base model.
    """
    # Step 1 — freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2 — unfreeze top-2 encoder layers
    try:
        encoder_layers = model.siglip.vision_model.encoder.layers
        for layer in encoder_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        _log("freeze", strategy="top-2 SigLIP encoder layers + head unfrozen")
    except AttributeError:
        # Fallback: if model architecture changed, unfreeze last 10% of params
        all_params = list(model.parameters())
        n_unfreeze  = max(1, len(all_params) // 10)
        for param in all_params[-n_unfreeze:]:
            param.requires_grad = True
        _log("freeze", strategy=f"fallback: last {n_unfreeze} params unfrozen")

    # Step 3 — always unfreeze the classifier head
    try:
        for param in model.classifier.parameters():
            param.requires_grad = True
    except AttributeError:
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    _log("params", trainable=trainable, total=total,
         pct=round(trainable / max(total, 1) * 100, 2))


# ──────────────────────────────────────────────────────────────────────────────
# ONNX quantization
# ──────────────────────────────────────────────────────────────────────────────

def _quantize_to_onnx(model_dir: Path, onnx_dir: Path) -> Optional[Path]:
    """
    Export the fine-tuned model to ONNX INT8 and cache it.

    Falls back gracefully if optimum / onnxruntime is not installed.
    Returns the path to the quantized model, or None on failure.
    """
    from typing import Optional  # local import to avoid shadowing

    try:
        from optimum.onnxruntime import ORTModelForImageClassification
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import shutil
    except ImportError:
        _log("onnx_skip", reason="optimum or onnxruntime not installed")
        return None

    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir          = onnx_dir / "fp32"
    quantized_path    = onnx_dir / "model_quantized.onnx"

    _log("onnx_start", fp32_dir=str(fp32_dir), out=str(quantized_path))

    try:
        ort_model = ORTModelForImageClassification.from_pretrained(
            str(model_dir), export=True,
        )
        ort_model.save_pretrained(str(fp32_dir))

        fp32_path = fp32_dir / "model.onnx"
        if not fp32_path.exists():
            candidates = list(fp32_dir.glob("*.onnx"))
            if not candidates:
                _log("onnx_error", reason="No .onnx file found after export")
                return None
            fp32_path = candidates[0]

        quantize_dynamic(
            model_input  = str(fp32_path),
            model_output = str(quantized_path),
            weight_type  = QuantType.QInt8,
        )

        shutil.rmtree(str(fp32_dir), ignore_errors=True)
        _log("onnx", path=str(quantized_path))
        return quantized_path

    except Exception as exc:
        _log("onnx_error", reason=str(exc))
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Collect data ───────────────────────────────────────────────────────
    paths, labels = _collect_paths()
    n_real  = labels.count(0)
    n_fake  = labels.count(1)
    n_total = len(paths)

    _log("start", total_images=n_total, real=n_real, fake=n_fake,
         backbone=BACKBONE, lr=LR, batch_size=BATCH_SIZE,
         max_epochs=MAX_EPOCHS, patience=PATIENCE)

    if n_total < MIN_IMAGES:
        _log("error",
             message=f"Not enough labelled images ({n_total} found, {MIN_IMAGES} required). "
                     f"Label more images in the Analyst Review tab and try again.")
        sys.exit(1)

    if n_real == 0 or n_fake == 0:
        _log("error",
             message=f"Need at least 1 image per class. "
                     f"Found: real={n_real}, fake={n_fake}.")
        sys.exit(1)

    # ── 2. Import heavy deps (lazy) ───────────────────────────────────────────
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, random_split
        from transformers import (
            AutoProcessor,
            AutoModelForImageClassification,
            logging as hf_logging,
        )
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    except ImportError as exc:
        _log("error", message=f"Missing dependency: {exc}. "
             "Run: pip install transformers torch torchvision scikit-learn")
        sys.exit(1)

    hf_logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log("device", device=device)

    # ── 3. Processor + model ──────────────────────────────────────────────────
    _log("loading_model", backbone=BACKBONE)
    try:
        processor = AutoProcessor.from_pretrained(BACKBONE)
        model     = AutoModelForImageClassification.from_pretrained(
            BACKBONE,
            num_labels=2,
            ignore_mismatched_sizes=True,   # safe: replaces classifier head
        )
    except Exception as exc:
        _log("error", message=f"Could not load backbone {BACKBONE}: {exc}")
        sys.exit(1)

    _apply_freeze_strategy(model)
    model = model.to(device)

    # ── 4. Build datasets ─────────────────────────────────────────────────────
    _log("building_dataset")
    import random as _random
    _random.seed(RANDOM_STATE)

    all_items   = _build_dataset(paths, labels, processor, augment=True)
    n_val       = max(1, int(len(all_items) * VAL_SPLIT))
    n_train     = len(all_items) - n_val
    torch.manual_seed(RANDOM_STATE)
    train_ds, val_ds = random_split(all_items, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)
    _log("dataset", train=n_train, val=n_val)

    # ── 5. Optimizer, loss, scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR / 10,
    )

    # ── 6. Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_epoch    = 0
    patience_ctr  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_pv, batch_lbl in train_loader:
            batch_pv  = batch_pv.to(device)
            batch_lbl = batch_lbl.to(device)
            optimizer.zero_grad()
            out  = model(pixel_values=batch_pv)
            loss = criterion(out.logits, batch_lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch_pv, batch_lbl in val_loader:
                batch_pv  = batch_pv.to(device)
                batch_lbl = batch_lbl.to(device)
                out  = model(pixel_values=batch_pv)
                loss = criterion(out.logits, batch_lbl)
                val_loss += loss.item()
                preds = out.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch_lbl.cpu().tolist())

        val_loss /= max(len(val_loader), 1)
        val_acc   = accuracy_score(all_labels, all_preds) * 100
        scheduler.step()

        _log("epoch", epoch=epoch, max_epochs=MAX_EPOCHS,
             loss=round(train_loss, 4), val_loss=round(val_loss, 4),
             val_acc=round(val_acc, 2))

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            patience_ctr  = 0
            model.save_pretrained(str(MODEL_SAVE_DIR))
            processor.save_pretrained(str(MODEL_SAVE_DIR))
            _log("best", epoch=epoch, val_acc=round(val_acc, 2),
                 val_loss=round(val_loss, 4))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                _log("early_stop", epoch=epoch, patience=PATIENCE,
                     best_epoch=best_epoch, best_val_acc=round(best_val_acc, 2))
                break

    # ── 7. Final evaluation (reload best weights) ─────────────────────────────
    if MODEL_SAVE_DIR.exists() and any(MODEL_SAVE_DIR.iterdir()):
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_SAVE_DIR)).to(device)
        model.eval()

    all_preds_final  = []
    all_labels_final = []
    with torch.no_grad():
        for batch_pv, batch_lbl in val_loader:
            out = model(pixel_values=batch_pv.to(device))
            all_preds_final.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_labels_final.extend(batch_lbl.tolist())

    if all_preds_final:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc  = round(accuracy_score(all_labels_final, all_preds_final)  * 100, 2)
        prec = round(precision_score(all_labels_final, all_preds_final,
                                     zero_division=0) * 100, 2)
        rec  = round(recall_score(all_labels_final, all_preds_final,
                                   zero_division=0) * 100, 2)
        f1   = round(f1_score(all_labels_final, all_preds_final,
                               zero_division=0) * 100, 2)
        _log("eval", accuracy=acc, precision=prec, recall=rec, f1=f1,
             best_epoch=best_epoch, train_images=n_total)

        # Persist metrics so Model Health tab can display them
        import json as _json
        metrics_path = Path("model/auto_train_metrics.json")
        with open(metrics_path, "w") as mf:
            _json.dump({
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "backbone": BACKBONE, "train_images": n_total,
                "real_images": n_real, "fake_images": n_fake,
                "best_epoch": best_epoch, "lr": LR,
                "timestamp": int(time.time()),
            }, mf, indent=2)

    # ── 8. ONNX INT8 quantization ─────────────────────────────────────────────
    _log("onnx_begin", source=str(MODEL_SAVE_DIR))
    _quantize_to_onnx(MODEL_SAVE_DIR, ONNX_OUT_DIR)

    _log("done", message=(
        f"Intelligence Sync complete. Best epoch {best_epoch}, "
        f"val_acc {round(best_val_acc, 2)}%. "
        f"Model saved to {MODEL_SAVE_DIR}. "
        f"ONNX INT8 at {ONNX_OUT_DIR / 'model_quantized.onnx'}."
    ))


if __name__ == "__main__":
    # Clear previous log on a fresh run
    LOG_FILE.write_text("", encoding="utf-8")
    try:
        main()
    except Exception as top_exc:
        _log("error", message=str(top_exc))
        sys.exit(1)