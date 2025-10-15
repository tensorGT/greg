"""Model registry and on-demand streaming downloader for greg."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
from platformdirs import user_cache_dir

DEFAULT_MODEL = "u2netp"


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and preprocessing spec for a segmentation ONNX model."""

    name: str
    filename: str
    url: str
    size_mb: int
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    description: str
    apply_sigmoid: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "u2netp": ModelSpec(
        name="u2netp",
        filename="u2netp.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        size_mb=5,
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        description="Default ultra-fast lightweight model (~5 MB). Great for icons, simple shapes, and bulk runs.",
    ),
    "silueta": ModelSpec(
        name="silueta",
        filename="silueta.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        size_mb=43,
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        description="Fast compact model (~43 MB) with cleaner edges than u2netp.",
    ),
    "u2net": ModelSpec(
        name="u2net",
        filename="u2net.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        size_mb=176,
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        description="Classic general-purpose segmentation model (~176 MB).",
    ),
    "isnet-general-use": ModelSpec(
        name="isnet-general-use",
        filename="isnet-general-use.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        size_mb=178,
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        description="High-resolution (1024px) model (~178 MB). Excellent for logos, products, and sharp edges.",
    ),
    "isnet-anime": ModelSpec(
        name="isnet-anime",
        filename="isnet-anime.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
        size_mb=176,
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        description="Specialized 1024px model (~176 MB) for anime, illustrations, and stickers.",
    ),
    "birefnet-general-lite": ModelSpec(
        name="birefnet-general-lite",
        filename="birefnet-general-lite.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        size_mb=214,
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        description="Fast BiRefNet variant (~214 MB) with strong hair and fine-edge detail.",
        apply_sigmoid=True,
    ),
    "birefnet-general": ModelSpec(
        name="birefnet-general",
        filename="birefnet-general.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
        size_mb=973,
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        description="Maximum-quality BiRefNet (~973 MB) for hair, fur, and complex scenes.",
        apply_sigmoid=True,
    ),
}


def get_models_dir() -> Path:
    """Return the local cache directory for ONNX models (respects GREG_MODELS_DIR)."""
    override = os.environ.get("GREG_MODELS_DIR")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = Path(user_cache_dir("greg", appauthor=False)) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_spec(model_name: str) -> ModelSpec:
    """Look up a ModelSpec by name or raise ValueError."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]


def get_model_path(model_name: str) -> Path:
    """Return expected local file path for a model."""
    spec = get_model_spec(model_name)
    return get_models_dir() / spec.filename


def is_model_cached(model_name: str) -> bool:
    """Return True if the model file exists and is non-empty."""
    path = get_model_path(model_name)
    return path.is_file() and path.stat().st_size > 0


ProgressCallback = Callable[[int, int], None]


def download_model(
    model_name: str,
    on_progress: ProgressCallback | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download an ONNX model atomically with optional progress callback (downloaded, total)."""
    spec = get_model_spec(model_name)
    dest = get_model_path(model_name)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    tmp_path = dest.with_suffix(".onnx.part")
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", spec.url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress is not None:
                            on_progress(downloaded, total)
        tmp_path.replace(dest)
        return dest
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
