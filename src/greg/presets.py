"""Named processing presets for common background-removal workflows."""

from __future__ import annotations

from dataclasses import dataclass

from greg.engine.models import DEFAULT_MODEL


@dataclass(frozen=True)
class PresetConfig:
    """Default parameters for a named preset."""

    name: str
    model: str
    defringe: float
    erode: int
    feather: float
    threshold: int | None
    trim: bool
    pad: int
    description: str


PRESETS: dict[str, PresetConfig] = {
    "fast": PresetConfig(
        name="fast",
        model=DEFAULT_MODEL,
        defringe=0.5,
        erode=0,
        feather=0.0,
        threshold=None,
        trim=False,
        pad=0,
        description="Ultra-fast mode using u2netp (~5 MB) with light defringing.",
    ),
    "logo": PresetConfig(
        name="logo",
        model="isnet-general-use",
        defringe=0.9,
        erode=1,
        feather=0.3,
        threshold=None,
        trim=True,
        pad=8,
        description="Crisp edges, strong halo defringing, auto-trim + 8px padding for logos & icons.",
    ),
    "portrait": PresetConfig(
        name="portrait",
        model="birefnet-general-lite",
        defringe=0.7,
        erode=0,
        feather=0.5,
        threshold=None,
        trim=False,
        pad=0,
        description="Soft hair/fur-friendly edges using BiRefNet.",
    ),
    "product": PresetConfig(
        name="product",
        model="isnet-general-use",
        defringe=0.85,
        erode=0,
        feather=0.3,
        threshold=None,
        trim=True,
        pad=16,
        description="Clean e-commerce cutouts with auto-trim and 16px margin.",
    ),
    "anime": PresetConfig(
        name="anime",
        model="isnet-anime",
        defringe=0.8,
        erode=0,
        feather=0.2,
        threshold=None,
        trim=False,
        pad=0,
        description="Tuned for illustrations, stickers, and anime artwork.",
    ),
}


def get_preset(name: str) -> PresetConfig:
    """Return a PresetConfig by name or raise ValueError."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return PRESETS[name]
