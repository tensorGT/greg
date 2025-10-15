"""Segmentation engine and model management."""

from greg.engine.models import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    ModelSpec,
    download_model,
    get_model_path,
    get_model_spec,
    get_models_dir,
    is_model_cached,
)
from greg.engine.session import get_available_providers, get_session, segment_image

__all__ = [
    "DEFAULT_MODEL",
    "MODEL_REGISTRY",
    "ModelSpec",
    "download_model",
    "get_available_providers",
    "get_model_path",
    "get_model_spec",
    "get_models_dir",
    "get_session",
    "is_model_cached",
    "segment_image",
]
