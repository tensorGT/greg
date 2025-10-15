"""ONNX Runtime session cache and image segmentation runner."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from greg.engine.models import ModelSpec, get_model_path, get_model_spec

_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, ort.InferenceSession] = {}

PREFERRED_PROVIDERS = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def get_available_providers() -> list[str]:
    """Return intersection of preferred providers and those installed in onnxruntime."""
    installed = set(ort.get_available_providers())
    selected = [p for p in PREFERRED_PROVIDERS if p in installed]
    return selected or ["CPUExecutionProvider"]


def get_session(model_name: str) -> ort.InferenceSession:
    """Load and cache an ONNX Runtime InferenceSession for the given model."""
    with _SESSION_LOCK:
        if model_name in _SESSIONS:
            return _SESSIONS[model_name]

        model_path = get_model_path(model_name)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model '{model_name}' is not downloaded yet at {model_path}."
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = get_available_providers()
        try:
            session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        except Exception:
            session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        _SESSIONS[model_name] = session
        return session


def _preprocess(image: Image.Image, spec: ModelSpec) -> np.ndarray:
    """Resize and normalize an RGB image into NCHW float32 tensor."""
    rgb = image.convert("RGB").resize(spec.input_size, Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.array(spec.mean, dtype=np.float32)
    std = np.array(spec.std, dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> NCHW
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def _postprocess_mask(output: np.ndarray, target_size: tuple[int, int], spec: ModelSpec) -> Image.Image:
    """Convert raw model output tensor into an L-mode alpha mask at target_size."""
    pred = np.squeeze(output).astype(np.float32)
    if pred.ndim > 2:
        pred = pred[0]

    if spec.apply_sigmoid and (pred.min() < 0.0 or pred.max() > 1.0):
        pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -30.0, 30.0)))

    ma, mi = float(pred.max()), float(pred.min())
    if ma - mi > 1e-6:
        pred = (pred - mi) / (ma - mi)
    else:
        pred = np.zeros_like(pred)

    mask_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_u8, mode="L")
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.Resampling.LANCZOS)
    return mask_img


def segment_image(image: Image.Image, model_name: str) -> Image.Image:
    """Run segmentation on `image` and return an L-mode alpha mask matching original dimensions."""
    spec = get_model_spec(model_name)
    session = get_session(model_name)
    input_name = session.get_inputs()[0].name
    tensor = _preprocess(image, spec)
    outputs: list[Any] = session.run(None, {input_name: tensor})
    # Most U2Net/ISNet models output main mask at index 0; BiRefNet outputs final stage at index -1.
    raw_out = outputs[-1] if spec.apply_sigmoid else outputs[0]
    return _postprocess_mask(np.asarray(raw_out), image.size, spec)
