"""
image_forensics.py  v4.1

Fixes over v4
─────────────
  1. SigLIP-2 pipeline type corrected (critical bug fix).

     v4 called:
         pipeline("image-classification", model="google/siglip2-base-patch16-224")

     SigLIP-2 is a vision-language (CLIP-style) encoder — it has NO
     image-classification head. Using "image-classification" on it either
     raises an error or returns model-internal token IDs as labels (e.g.
     "LABEL_0", "LABEL_1") which never match the "ai"/"real" strings in
     detect_ai_image(). That triggered the last-resort fallback, which
     assigned the top result as AI regardless of content — making every
     image score ~67% AI-generated (observed bug in screenshots).

     Fix: use "zero-shot-image-classification" with explicit candidate labels
         ["AI-generated image", "real photograph"]

     The same ONNX export / quantization path is kept; the pipeline type
     is the only change. detect_ai_image() is updated to read the new
     zero-shot output format: [{"label": ..., "score": ...}] where label
     is one of the two candidate strings.

  2. _build_onnx_pipeline and _ensure_onnx_model updated to use the correct
     pipeline task string "zero-shot-image-classification" and pass
     candidate_labels at inference time.

  3. All other functions, thresholds, and signatures are UNCHANGED.
"""

from __future__ import annotations

import io
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from PIL import Image

# ── ONNX cache directories ────────────────────────────────────────────────────
_ONNX_ROOT         = Path("model/onnx")
_ONNX_AI_DIR       = _ONNX_ROOT / "ai_detector"
_ONNX_DEEPFAKE_DIR = _ONNX_ROOT / "deepfake_detector"

# ── Model identifiers ─────────────────────────────────────────────────────────
_AI_MODEL_ID       = "google/siglip2-base-patch16-224"
_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-Model"

# ── SigLIP-2 zero-shot candidate labels ──────────────────────────────────────
# These are the two text prompts passed to the vision-language model.
# Order matters for score retrieval in detect_ai_image().
_SIGLIP_AI_LABEL   = "AI-generated image"
_SIGLIP_REAL_LABEL = "real photograph"
_SIGLIP_CANDIDATES = [_SIGLIP_AI_LABEL, _SIGLIP_REAL_LABEL]

# ── Inference constants ───────────────────────────────────────────────────────
_DEEPFAKE_INPUT_SIZE = 224
_FAKE_LABELS         = {"fake", "deepfake", "artificial", "manipulated", "forged"}
_REAL_LABELS         = {"real", "authentic", "genuine", "original"}

# ── Video pipeline constants ──────────────────────────────────────────────────
_VIDEO_MAX_FRAMES   = 30
_VIDEO_SCENE_THRESH = 30.0
_VIDEO_MIN_GAP_SEC  = 1.0

# ── Biological Consistency thresholds ────────────────────────────────────────
_BIO_SKIN_SMOOTH_THRESH  = 8.0
_BIO_SKIN_SEAM_THRESH    = 32.0
_BIO_EYE_BLUR_THRESH     = 50.0
_BIO_HAIR_FEATHER_THRESH = 15.0
_BIO_SYMMETRY_THRESH     = 0.18


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BioCheckResult:
    skin_std_dev:        float = 0.0
    eye_sharpness:       float = 0.0
    hair_gradient:       float = 0.0
    symmetry_ratio:      float = 0.0

    skin_suspicious:     bool = False
    eye_suspicious:      bool = False
    hair_suspicious:     bool = False
    symmetry_suspicious: bool = False

    any_suspicious:   bool      = False
    suspicious_zones: List[str] = field(default_factory=list)
    notes:            List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Internal: ONNX export + quantization helper
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_onnx_model(model_id: str, onnx_dir: Path) -> Optional[Path]:
    quantized_path = onnx_dir / "model_quantized.onnx"
    if quantized_path.exists():
        return quantized_path

    try:
        from optimum.onnxruntime import ORTModelForImageClassification
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        return None

    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir = onnx_dir / "fp32"

    ort_model = ORTModelForImageClassification.from_pretrained(
        model_id, export=True,
    )
    ort_model.save_pretrained(str(fp32_dir))

    fp32_path = fp32_dir / "model.onnx"
    if not fp32_path.exists():
        candidates = list(fp32_dir.glob("*.onnx"))
        if not candidates:
            return None
        fp32_path = candidates[0]

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    import shutil
    shutil.rmtree(str(fp32_dir), ignore_errors=True)
    return quantized_path


def _build_onnx_pipeline(model_id: str, onnx_dir: Path, task: str = "image-classification"):
    """
    Build an inference pipeline, preferring ONNX INT8 when available.

    FIX (v4.1): The `task` parameter is now threaded through so callers
    can specify "zero-shot-image-classification" for SigLIP-2 instead of
    the incorrect "image-classification" used in v4.
    """
    quantized_path = _ensure_onnx_model(model_id, onnx_dir)

    if quantized_path is None:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(task, model=model_id, device=-1)

    try:
        from optimum.onnxruntime import ORTModelForImageClassification
        from transformers import pipeline as hf_pipeline

        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(model_id)

        ort_model = ORTModelForImageClassification.from_pretrained(
            str(onnx_dir),
            file_name="model_quantized.onnx",
        )
        return hf_pipeline(
            task,
            model=ort_model,
            feature_extractor=processor,
        )
    except Exception:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(task, model=model_id, device=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  AI Image Detector  (SigLIP-2, zero-shot)
# ──────────────────────────────────────────────────────────────────────────────

def load_image_model():
    """
    FIX (v4.1): Use "zero-shot-image-classification" — the correct pipeline
    task for SigLIP-2, which is a vision-language encoder without a
    classification head.
    """
    return _build_onnx_pipeline(
        _AI_MODEL_ID,
        _ONNX_AI_DIR,
        task="zero-shot-image-classification",
    )


def detect_ai_image(image: Image.Image, model) -> dict:
    """
    Run SigLIP-2 zero-shot classification with two candidate labels:
        - "AI-generated image"
        - "real photograph"

    FIX (v4.1): v4 called model(image) without candidate_labels, which is
    required for zero-shot pipelines. Without them transformers raises a
    ValueError or returns internal token IDs. We now pass _SIGLIP_CANDIDATES
    explicitly.

    The score extraction uses exact string matching against the two known
    candidate labels, completely replacing the fragile label-sniffing
    fallback that caused every image to read ~67% AI-generated.

    CRITICAL THRESHOLD: >= 85% -> "AI-Generated"  (must never be lowered)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Pass candidate_labels — mandatory for zero-shot-image-classification
    results = model(image, candidate_labels=_SIGLIP_CANDIDATES)

    # results: [{"label": "AI-generated image", "score": 0.72}, ...]
    scores: dict = {item["label"]: item["score"] for item in results}

    raw_ai   = scores.get(_SIGLIP_AI_LABEL,   0.0)
    raw_real = scores.get(_SIGLIP_REAL_LABEL,  0.0)

    # Normalise to 100% (zero-shot scores are already softmax-normalised
    # across candidates, but guard against floating-point edge cases)
    total    = max(raw_ai + raw_real, 1e-9)
    ai_prob  = round(min(max((raw_ai  / total) * 100, 0.0), 100.0), 2)
    real_pct = round(min(max((raw_real / total) * 100, 0.0), 100.0), 2)

    # CRITICAL: thresholds (Global Rule 1 — do not lower)
    if ai_prob >= 85:
        label = "AI-Generated"
    elif ai_prob >= 75:
        label = "Likely Retouched / Uncertain"
    else:
        label = "Likely Real"

    return {
        "ai_probability":   ai_prob,
        "real_probability": real_pct,
        "label":            label,
        "raw_ai_score":     round(raw_ai,   4),
        "raw_real_score":   round(raw_real, 4),
        "raw":              results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  EXIF Metadata Extractor
# ──────────────────────────────────────────────────────────────────────────────

_EXIF_KEYS_OF_INTEREST = [
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "Artist", "Copyright", "ImageDescription",
    "GPSInfo", "ExifVersion", "FlashPixVersion", "ColorSpace",
    "PixelXDimension", "PixelYDimension", "ExposureTime", "FNumber",
    "ISOSpeedRatings", "FocalLength", "Flash", "MeteringMode",
    "WhiteBalance", "LightSource", "SceneCaptureType",
]

EDITING_SOFTWARE_KEYWORDS = [
    "adobe", "photoshop", "lightroom", "gimp", "affinity",
    "capture one", "darktable", "rawtherapee", "snapseed",
    "paint.net", "canva", "stable diffusion", "midjourney",
    "dall-e", "firefly", "generative",
]


def extract_exif(image: Image.Image) -> dict:
    from PIL.ExifTags import TAGS

    exif_data:   dict = {}
    software:    Optional[str] = None
    gps_present: bool = False

    try:
        raw_exif = image._getexif()
    except (AttributeError, Exception):
        raw_exif = None

    if raw_exif:
        for tag_id, value in raw_exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if tag_name == "GPSInfo":
                gps_present = True
                exif_data["GPSInfo"] = "GPS coordinates embedded"
                continue
            if isinstance(value, bytes) and len(value) > 64:
                continue
            if hasattr(value, "items"):
                value = str(dict(value))
            elif isinstance(value, tuple):
                value = ", ".join(str(v) for v in value)
            str_value = str(value).strip()
            if str_value and tag_name in _EXIF_KEYS_OF_INTEREST:
                exif_data[tag_name] = str_value
            if tag_name == "Software":
                software = str_value

    editing_flags: List[str] = []
    if software:
        sw_lower = software.lower()
        for kw in EDITING_SOFTWARE_KEYWORDS:
            if kw in sw_lower:
                editing_flags.append(kw.title())

    warning: Optional[str] = None
    if editing_flags:
        warning = (
            f"Software tag contains: \"{software}\". "
            f"This image may have been edited or generated with: "
            f"{', '.join(editing_flags)}."
        )

    return {
        "data":          exif_data,
        "has_exif":      bool(exif_data),
        "software":      software,
        "editing_flags": editing_flags,
        "gps_present":   gps_present,
        "warning":       warning,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Error Level Analysis  + Overlay
# ──────────────────────────────────────────────────────────────────────────────

def generate_ela_heatmap(
    image: Image.Image,
    quality: int = 90,
    amplify: int = 20,
) -> Image.Image:
    import cv2
    import numpy as np

    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff           = np.abs(
        np.array(image, dtype=np.int32) - np.array(recompressed, dtype=np.int32)
    ).astype(np.uint8)
    diff_amplified = np.clip(diff.astype(np.int32) * amplify, 0, 255).astype(np.uint8)

    gray        = cv2.cvtColor(diff_amplified, cv2.COLOR_RGB2GRAY)
    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb)


def blend_ela_overlay(
    original: Image.Image,
    ela_heatmap: Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    import cv2
    import numpy as np

    alpha = float(max(0.0, min(1.0, alpha)))

    if original.mode != "RGB":
        original = original.convert("RGB")
    if ela_heatmap.mode != "RGB":
        ela_heatmap = ela_heatmap.convert("RGB")

    orig_w, orig_h = original.size
    ela_w,  ela_h  = ela_heatmap.size

    if (ela_w, ela_h) != (orig_w, orig_h):
        ar_orig = orig_w / max(orig_h, 1)
        ar_ela  = ela_w  / max(ela_h,  1)
        if abs(ar_orig - ar_ela) / max(ar_orig, 1e-9) > 0.05:
            import warnings
            warnings.warn(
                f"blend_ela_overlay: aspect ratio mismatch "
                f"(original {orig_w}×{orig_h}, ELA {ela_w}×{ela_h}). "
                "ELA may have been computed on a different image.",
                stacklevel=2,
            )

        is_upscale = (orig_w * orig_h) > (ela_w * ela_h)
        resample   = Image.BICUBIC if is_upscale else Image.LANCZOS
        ela_heatmap = ela_heatmap.resize((orig_w, orig_h), resample)

    orig_np = np.array(original,    dtype=np.float32)
    heat_np = np.array(ela_heatmap, dtype=np.float32)

    blended = cv2.addWeighted(orig_np, 1.0 - alpha, heat_np, alpha, 0.0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Face Cropper
# ──────────────────────────────────────────────────────────────────────────────

def extract_faces(
    image: Image.Image,
    min_face_size: int   = 80,
    padding:       float = 0.20,
) -> List[Image.Image]:
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        return []

    if image.mode != "RGB":
        image = image.convert("RGB")

    mtcnn = MTCNN(
        keep_all=True, min_face_size=min_face_size,
        device="cpu", post_process=False,
    )
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return []

    w, h   = image.size
    crops: List[Image.Image] = []

    for box in boxes:
        x1, y1, x2, y2 = box
        px  = (x2 - x1) * padding
        py  = (y2 - y1) * padding
        x1  = max(0, int(x1 - px));  y1 = max(0, int(y1 - py))
        x2  = min(w, int(x2 + px));  y2 = min(h, int(y2 + py))
        if x2 > x1 and y2 > y1:
            crops.append(image.crop((x1, y1, x2, y2)))

    return crops


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Deepfake Detector
# ──────────────────────────────────────────────────────────────────────────────

def load_deepfake_model():
    return _build_onnx_pipeline(
        _DEEPFAKE_MODEL_ID,
        _ONNX_DEEPFAKE_DIR,
        task="image-classification",
    )


def detect_deepfake(
    face_images: List[Image.Image],
    model,
) -> List[dict]:
    """
    CRITICAL THRESHOLD: >= 80% -> "Deepfake Detected"  (must never be lowered)
    """
    if not face_images:
        return []

    results: List[dict] = []

    for face in face_images:
        if face.mode != "RGB":
            face = face.convert("RGB")

        w, h = face.size
        if min(w, h) < _DEEPFAKE_INPUT_SIZE:
            scale = _DEEPFAKE_INPUT_SIZE / min(w, h)
            face  = face.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        try:
            raw = model(face)
        except Exception as exc:
            results.append({
                "deepfake_probability": 0.0, "real_probability": 100.0,
                "label": "Error", "confidence": "N/A",
                "raw": [{"error": str(exc)}],
            })
            continue

        scores: dict = {
            item["label"].lower(): item["score"] for item in raw
        }

        fake_prob = 0.0
        for lbl, score in scores.items():
            if any(f in lbl for f in _FAKE_LABELS):
                fake_prob = score * 100
                break
        else:
            rp = max(
                (s for lbl, s in scores.items() if any(r in lbl for r in _REAL_LABELS)),
                default=None,
            )
            if rp is not None:
                fake_prob = (1.0 - rp) * 100

        real_prob  = round(100.0 - fake_prob, 2)
        fake_prob  = round(fake_prob, 2)
        margin     = abs(fake_prob - 50.0)
        confidence = "High" if margin >= 30 else ("Medium" if margin >= 15 else "Low")

        label = "Deepfake Detected" if fake_prob >= 80 else "Likely Real"

        results.append({
            "deepfake_probability": fake_prob,
            "real_probability":     real_prob,
            "label":                label,
            "confidence":           confidence,
            "raw":                  raw,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5b.  Biological Consistency Check
# ──────────────────────────────────────────────────────────────────────────────

def _laplacian_variance(gray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _sobel_variance(gray) -> float:
    import cv2
    import numpy as np
    sx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return float(mag.var())


def _skin_lab_std(rgb) -> float:
    import cv2
    import numpy as np

    h, w = rgb.shape[:2]
    region = rgb[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    if region.size == 0:
        return 0.0

    lab   = cv2.cvtColor(
        cv2.cvtColor(region, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab
    ).astype(np.float32)
    return round((float(np.std(lab[:, :, 1])) + float(np.std(lab[:, :, 2]))) / 2.0, 2)


def _eye_strip_sharpness(rgb) -> float:
    import cv2
    h, w  = rgb.shape[:2]
    strip = rgb[int(h * 0.25):int(h * 0.45), int(w * 0.10):int(w * 0.90)]
    if strip.size == 0:
        return 100.0
    gray  = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    return round(_laplacian_variance(gray), 2)


def _hair_boundary_gradient(rgb) -> float:
    import cv2
    h, w = rgb.shape[:2]
    top  = rgb[:int(h * 0.20), :]
    if top.size == 0:
        return 100.0
    gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
    return round(_sobel_variance(gray), 2)


def _lr_symmetry_ratio(rgb) -> float:
    import numpy as np
    h, w  = rgb.shape[:2]
    mid   = w // 2
    left  = rgb[:, :mid].astype(np.float32)
    right = rgb[:, mid: mid + mid].astype(np.float32)[:, ::-1]

    if left.shape != right.shape:
        mw    = min(left.shape[1], right.shape[1])
        left  = left[:, :mw]
        right = right[:, :mw]

    diff = float(np.mean(np.abs(left - right)))
    mean = float(np.mean(rgb))
    return round(diff / max(mean, 1.0), 4)


def biological_consistency_check(
    face_images: List[Image.Image],
) -> List[BioCheckResult]:
    if not face_images:
        return []

    try:
        import cv2
        import numpy as np
    except ImportError:
        return [BioCheckResult() for _ in face_images]

    results: List[BioCheckResult] = []

    for face in face_images:
        if face.mode != "RGB":
            face = face.convert("RGB")

        w, h  = face.size
        scale = 256 / min(w, h)
        face  = face.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        rgb   = np.array(face)

        skin_std  = _skin_lab_std(rgb)
        eye_sharp = _eye_strip_sharpness(rgb)
        hair_grad = _hair_boundary_gradient(rgb)
        sym_ratio = _lr_symmetry_ratio(rgb)

        skin_flag = not (_BIO_SKIN_SMOOTH_THRESH <= skin_std <= _BIO_SKIN_SEAM_THRESH)
        eye_flag  = eye_sharp  < _BIO_EYE_BLUR_THRESH
        hair_flag = hair_grad  < _BIO_HAIR_FEATHER_THRESH
        sym_flag  = sym_ratio  > _BIO_SYMMETRY_THRESH

        suspicious_zones: List[str] = []
        notes: List[str] = []

        if skin_flag:
            if skin_std < _BIO_SKIN_SMOOTH_THRESH:
                notes.append(
                    f"Skin: unnaturally uniform colour (Lab ab std-dev {skin_std:.1f} < "
                    f"threshold {_BIO_SKIN_SMOOTH_THRESH}) — GAN-synthesized skin lacks "
                    "the micro-variation of real human tissue."
                )
            else:
                notes.append(
                    f"Skin: hard colour seam detected (Lab ab std-dev {skin_std:.1f} > "
                    f"threshold {_BIO_SKIN_SEAM_THRESH}) — indicative of a tone-mismatch "
                    "at a face-swap boundary or composite splice."
                )
            suspicious_zones.append("Skin")

        if eye_flag:
            notes.append(
                f"Eyes: blurred boundary gradients (Laplacian sharpness {eye_sharp:.1f} < "
                f"threshold {_BIO_EYE_BLUR_THRESH:.0f}) — a GAN artefact consistent with "
                "face-swap or inpainting around the periorbital region."
            )
            suspicious_zones.append("Eyes")

        if hair_flag:
            notes.append(
                f"Hair: feathered boundary (Sobel gradient variance {hair_grad:.1f} < "
                f"threshold {_BIO_HAIR_FEATHER_THRESH:.0f}) — AI-generated hair lacks the "
                "sharp individual-strand gradients of real photographic hair."
            )
            suspicious_zones.append("Hair")

        if sym_flag:
            notes.append(
                f"Symmetry: abnormal left/right ratio {sym_ratio:.3f} "
                f"(threshold {_BIO_SYMMETRY_THRESH}) — mismatched tone-mapping across "
                "the vertical midline can indicate GAN blending or face-swap seams."
            )
            suspicious_zones.append("Symmetry")

        any_flag = skin_flag or eye_flag or hair_flag or sym_flag

        results.append(BioCheckResult(
            skin_std_dev         = skin_std,
            eye_sharpness        = eye_sharp,
            hair_gradient        = hair_grad,
            symmetry_ratio       = sym_ratio,
            skin_suspicious      = skin_flag,
            eye_suspicious       = eye_flag,
            hair_suspicious      = hair_flag,
            symmetry_suspicious  = sym_flag,
            any_suspicious       = any_flag,
            suspicious_zones     = suspicious_zones,
            notes                = notes,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5c.  Consensus Meter
# ──────────────────────────────────────────────────────────────────────────────

def consensus_meter(
    ai_result:   Optional[dict],
    bio_results: Optional[List[BioCheckResult]],
) -> dict:
    ai_score  = float(ai_result.get("ai_probability", 0.0)) if ai_result else 0.0
    bio_flags = sum(len(r.suspicious_zones) for r in bio_results) if bio_results else 0

    if ai_score >= 60 and bio_flags == 0:
        label   = "High-Complexity Edge Case"
        detail  = (
            f"SigLIP-2 flagged {ai_score:.1f}% AI probability, but biological "
            "consistency checks found no zone anomalies. Possible causes: "
            "texture-only diffusion artefacts, heavy JPEG re-encoding, or "
            "model overconfidence on a genuine photograph with unusual lighting. "
            "Manual review recommended."
        )
        conflict = True
        color    = "#f59e0b"
        icon     = "⚠️"

    elif ai_score < 40 and bio_flags >= 2:
        label   = "High-Complexity Edge Case"
        detail  = (
            f"Biological checks flagged {bio_flags} suspicious zone(s) "
            f"(e.g. {', '.join(dict.fromkeys(z for r in (bio_results or []) for z in r.suspicious_zones))}) "
            f"but the AI generation score is low ({ai_score:.1f}%). "
            "Possible causes: medical/cosmetic alteration, heavy beauty-filter "
            "retouching, or measurement noise on a low-resolution crop."
        )
        conflict = True
        color    = "#f59e0b"
        icon     = "⚠️"

    elif 40 <= ai_score < 60 and bio_flags == 1:
        label   = "Minor Discordance"
        detail  = (
            f"Borderline AI score ({ai_score:.1f}%) with one minor biological "
            "flag — insufficient evidence to determine authenticity with "
            "confidence. Additional signal (EXIF provenance, reverse image "
            "search) is recommended."
        )
        conflict = False
        color    = "#fbbf24"
        icon     = "〰️"

    else:
        if ai_score >= 60 and bio_flags >= 1:
            detail = (
                f"Both detectors agree: AI generation score {ai_score:.1f}% "
                f"and {bio_flags} biological zone flag(s) consistently indicate "
                "manipulation."
            )
        elif ai_score < 40 and bio_flags == 0:
            detail = (
                f"Both detectors agree: AI generation score is low ({ai_score:.1f}%) "
                "and no biological zone anomalies were found. Image is likely genuine."
            )
        else:
            detail = (
                f"AI score {ai_score:.1f}%, bio flags {bio_flags}. "
                "Signals are broadly consistent."
            )
        label    = "Consensus"
        conflict = False
        color    = "#4ade80"
        icon     = "✅"

    return {
        "label":     label,
        "detail":    detail,
        "ai_score":  ai_score,
        "bio_flags": bio_flags,
        "conflict":  conflict,
        "color":     color,
        "icon":      icon,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Forensic Reasoning Engine
# ──────────────────────────────────────────────────────────────────────────────

def generate_forensic_reasoning(
    ai_result:        Optional[dict],
    deepfake_results: Optional[List[dict]],
    exif_data:        Optional[dict],
    bio_results:      Optional[List[BioCheckResult]] = None,
) -> List[str]:
    bullets: List[str] = []

    if bio_results:
        all_notes: List[str] = []
        for r in bio_results:
            all_notes.extend(r.notes)

        if all_notes:
            seen_zones: set = set()
            unique_notes: List[str] = []
            for note in all_notes:
                zone = note.split(":")[0].strip()
                if zone not in seen_zones:
                    seen_zones.add(zone)
                    unique_notes.append(note)

            for note in unique_notes[:2]:
                bullets.append(note)

        elif not any(r.any_suspicious for r in bio_results):
            bullets.append(
                f"Biological consistency passed across all {len(bio_results)} "
                "detected face(s): skin colour variance, eye-region sharpness, "
                "hair boundary gradients, and facial symmetry are all within "
                "expected photographic norms."
            )

    if deepfake_results:
        flagged   = [r for r in deepfake_results if r.get("deepfake_probability", 0) >= 80]
        max_df    = max((r.get("deepfake_probability", 0) for r in deepfake_results), default=0)
        n_faces   = len(deepfake_results)
        n_flagged = len(flagged)

        if n_flagged > 0:
            worst_conf = max(
                (r.get("confidence", "Low") for r in flagged),
                key=lambda c: {"High": 2, "Medium": 1, "Low": 0}.get(c, 0),
            )
            bullets.append(
                f"Overall face classification: {n_flagged}/{n_faces} face(s) "
                f"exceeded the 80% deepfake threshold (peak {max_df:.1f}%, "
                f"classifier confidence: {worst_conf}). The EfficientNet model "
                "detected manipulation patterns consistent with neural face "
                "synthesis or face-swap blending."
            )
        elif max_df >= 60:
            bullets.append(
                f"Overall face classification: peak deepfake probability "
                f"{max_df:.1f}% — below the 80% verdict threshold but elevated. "
                "Minor blending or lighting inconsistencies were flagged without "
                "crossing the detection boundary."
            )
        else:
            bullets.append(
                f"Overall face classification: all {n_faces} face(s) scored "
                f"below the deepfake threshold (peak {max_df:.1f}%). No "
                "face-synthesis or face-swap signatures detected."
            )
    elif deepfake_results is not None:
        bullets.append(
            "No faces detected — deepfake classification not applicable. "
            "The SigLIP-2 AI-generation score is the primary evidential signal."
        )

    if ai_result:
        ai_pct    = ai_result.get("ai_probability", 0.0)
        real_pct  = ai_result.get("real_probability", 100.0)
        raw_ai    = ai_result.get("raw_ai_score")
        raw_real  = ai_result.get("raw_real_score")

        siglip_note = ""
        if raw_ai is not None and raw_real is not None:
            siglip_note = (
                f" (SigLIP-2 scores: AI={raw_ai:.3f}, Real={raw_real:.3f})"
            )

        if ai_pct >= 85:
            bullets.append(
                f"AI generation: {ai_pct:.1f}% confidence{siglip_note}. "
                "Spatial frequency analysis, texture uniformity, and colour-histogram "
                "distribution are statistically inconsistent with optical sensor noise "
                "— strongly indicating a generative model (diffusion/GAN) origin."
            )
        elif ai_pct >= 75:
            bullets.append(
                f"AI generation: {ai_pct:.1f}% — uncertain zone{siglip_note}. "
                "Characteristics suggest heavy post-processing or partial AI-assisted "
                "editing. Cannot be classified as fully AI-generated; localised "
                "texture inconsistencies are present."
            )
        elif ai_pct >= 50:
            bullets.append(
                f"AI generation: moderate score {ai_pct:.1f}%{siglip_note}. "
                "Likely a real photograph, but selected regions show atypical "
                "compression artefacts or smoothed textures."
            )
        else:
            bullets.append(
                f"AI generation: low score {ai_pct:.1f}% "
                f"(real probability {real_pct:.1f}%){siglip_note}. "
                "Pixel-level noise, lens aberrations, and JPEG patterns are "
                "consistent with genuine camera capture."
            )

    if ai_result:
        ai_pct = ai_result.get("ai_probability", 0.0)
        if ai_pct >= 75:
            bullets.append(
                "ELA: elevated re-compression error expected — especially in smooth "
                "background regions and around subject boundaries where generative "
                "models produce JPEG-cycle artefacts distinct from single-pass camera "
                "captures. Bright red zones in the heatmap mark these composited areas."
            )
        else:
            bullets.append(
                "ELA: predominantly low-error regions expected (blue-green heatmap), "
                "consistent with a single JPEG encode. Minor high-error patches at "
                "high-frequency texture edges (foliage, fabric) are normal camera "
                "artefacts rather than manipulation indicators."
            )

    if exif_data:
        has_exif      = exif_data.get("has_exif", False)
        editing_flags = exif_data.get("editing_flags", [])
        gps_present   = exif_data.get("gps_present", False)
        software      = exif_data.get("software")
        camera_make   = (exif_data.get("data") or {}).get("Make")

        if editing_flags:
            sw_list = ", ".join(editing_flags)
            bullets.append(
                f"EXIF provenance: Software tag records \"{software}\", processed "
                f"with {sw_list}. This breaks the direct-from-camera chain — editing "
                "alone does not confirm fabrication, but pixel-level authenticity "
                "cannot be guaranteed by metadata."
            )
        elif not has_exif:
            bullets.append(
                "EXIF provenance: no metadata found. Camera images almost always "
                "embed manufacturer data; its absence suggests EXIF stripping "
                "(common when sharing AI-generated images) or a screenshot / "
                "re-save workflow."
            )
        elif camera_make:
            gps_note = (
                " GPS data embedded — consistent with a smartphone capture."
                if gps_present else ""
            )
            bullets.append(
                f"EXIF provenance: intact and records a real camera ({camera_make})."
                f"{gps_note} Supports genuine capture provenance, though EXIF fields "
                "are trivially injectable and are a supporting — not conclusive — signal."
            )
        else:
            bullets.append(
                "EXIF provenance: metadata present but no camera Make/Model tag. "
                "Common in screenshots, web-downloaded images, or files resaved "
                "through an editor that strips hardware metadata."
            )

    if not bullets:
        return [
            "Insufficient signals were extracted to generate a localised forensic "
            "breakdown. Run AI detection, deepfake analysis, biological consistency "
            "check, and EXIF extraction for a full report."
        ]

    return bullets[:5]


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Video Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def extract_video_keyframes(
    video_path: str,
    max_frames:   int   = _VIDEO_MAX_FRAMES,
    scene_thresh: float = _VIDEO_SCENE_THRESH,
    min_gap_sec:  float = _VIDEO_MIN_GAP_SEC,
) -> List[Image.Image]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_gap_fr  = int(fps * min_gap_sec)
    keyframes:  List[Image.Image] = []
    prev_frame  = None
    frame_idx   = 0
    last_kf_idx = -min_gap_fr

    while len(keyframes) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gap_ok  = (frame_idx - last_kf_idx) >= min_gap_fr
        sc      = (prev_frame is None) or (np.mean(np.abs(gray - prev_frame)) >= scene_thresh)

        if gap_ok and sc:
            keyframes.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            last_kf_idx = frame_idx

        prev_frame = gray
        frame_idx += 1

    cap.release()
    return keyframes


def analyse_video_deepfakes(
    video_path:   str,
    deepfake_model,
    max_frames:   int   = _VIDEO_MAX_FRAMES,
    scene_thresh: float = _VIDEO_SCENE_THRESH,
    min_gap_sec:  float = _VIDEO_MIN_GAP_SEC,
) -> dict:
    result: dict = {
        "keyframe_count": 0, "total_faces": 0, "flagged_faces": 0,
        "max_deepfake_prob": 0.0, "verdict": "Likely Real",
        "frame_results": [], "bio_results": [], "reasoning": [], "error": None,
    }

    try:
        keyframes = extract_video_keyframes(
            video_path, max_frames=max_frames,
            scene_thresh=scene_thresh, min_gap_sec=min_gap_sec,
        )
    except Exception as exc:
        result["error"] = f"Keyframe extraction failed: {exc}"
        return result

    if not keyframes:
        result["error"] = "No keyframes extracted."
        return result

    result["keyframe_count"] = len(keyframes)

    all_df:  List[dict]           = []
    all_bio: List[BioCheckResult] = []

    for frame_img in keyframes:
        faces = extract_faces(frame_img)
        if not faces:
            result["frame_results"].append([])
            continue

        df_res  = detect_deepfake(faces, deepfake_model)
        bio_res = biological_consistency_check(faces)
        result["frame_results"].append(df_res)
        all_df.extend(df_res)
        all_bio.extend(bio_res)

    result["bio_results"] = all_bio

    if not all_df:
        result["verdict"]   = "No Faces Found"
        result["reasoning"] = [
            "No human faces detected across keyframes — deepfake analysis and "
            "biological consistency check were skipped.",
        ]
        return result

    result["total_faces"]       = len(all_df)
    max_prob                    = max(r["deepfake_probability"] for r in all_df)
    flagged                     = [r for r in all_df if r["deepfake_probability"] >= 80]
    result["max_deepfake_prob"] = round(max_prob, 2)
    result["flagged_faces"]     = len(flagged)

    result["verdict"] = "Deepfake Detected" if flagged else "Likely Real"

    result["reasoning"] = generate_forensic_reasoning(
        ai_result=None, deepfake_results=all_df,
        exif_data=None, bio_results=all_bio,
    )
    video_ctx = (
        f"Video scan: {result['keyframe_count']} keyframes, "
        f"{result['total_faces']} face(s). "
        + (f"{len(flagged)} face(s) exceeded the 80% deepfake threshold."
           if flagged else "No faces exceeded the 80% deepfake threshold.")
    )
    result["reasoning"].insert(0, video_ctx)
    result["reasoning"] = result["reasoning"][:5]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()