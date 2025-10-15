"""Fast NumPy/Pillow alpha refinement, color defringing, and canvas trimming."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def refine_alpha(
    mask: Image.Image,
    *,
    erode: int = 0,
    feather: float = 0.0,
    threshold: int | None = None,
) -> Image.Image:
    """Refine an L-mode alpha mask with optional thresholding, erosion/dilation, and feathering.

    - `erode > 0` shrinks the mask (cuts edge halos); `erode < 0` dilates it.
    - `feather > 0` applies a Gaussian blur of radius `feather`.
    - `threshold` (0..255) snaps soft edges to a crisp binary-like edge before feathering.
    """
    out = mask.convert("L")

    if threshold is not None:
        arr = np.asarray(out, dtype=np.uint8)
        arr = np.where(arr >= threshold, 255, 0).astype(np.uint8)
        out = Image.fromarray(arr, mode="L")

    if erode > 0:
        size = 2 * erode + 1
        out = out.filter(ImageFilter.MinFilter(size=size))
    elif erode < 0:
        size = 2 * abs(erode) + 1
        out = out.filter(ImageFilter.MaxFilter(size=size))

    if feather > 0.0:
        out = out.filter(ImageFilter.GaussianBlur(radius=feather))

    return out


def defringe_rgba(image: Image.Image, strength: float = 0.8, radius: int = 3) -> Image.Image:
    """Reduce background color bleed on semi-transparent edge pixels.

    Replaces edge RGB values with nearby solid-foreground RGB values weighted by `strength` (0..1).
    """
    if strength <= 0.0:
        return image

    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    alpha = arr[..., 3]

    # Pixels that are semi-transparent or near the boundary
    edge_mask = (alpha > 0) & (alpha < 245)
    if not np.any(edge_mask):
        return rgba

    # Build a foreground-only RGB image where non-solid pixels are filled via local max/blur propagation
    solid = alpha >= 245
    if not np.any(solid):
        return rgba

    rgb_img = Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")
    # Iteratively dilate solid foreground colors into edge region using MaxFilter/Blur approximation
    fg_arr = arr[..., :3].copy()
    fg_arr[~solid] = 0.0
    weight = solid.astype(np.float32)

    fg_pil = Image.fromarray(np.clip(fg_arr, 0, 255).astype(np.uint8), mode="RGB")
    w_pil = Image.fromarray(np.clip(weight * 255.0, 0, 255).astype(np.uint8), mode="L")

    blur_r = max(1, radius)
    blurred_fg = np.asarray(fg_pil.filter(ImageFilter.GaussianBlur(radius=blur_r)), dtype=np.float32)
    blurred_w = np.asarray(w_pil.filter(ImageFilter.GaussianBlur(radius=blur_r)), dtype=np.float32) / 255.0

    valid = blurred_w > 1e-4
    estimated_fg = arr[..., :3].copy()
    estimated_fg[valid] = np.clip(blurred_fg[valid] / blurred_w[valid, None], 0.0, 255.0)

    blend = np.clip(strength, 0.0, 1.0)
    out = arr.copy()
    out[edge_mask, :3] = (1.0 - blend) * arr[edge_mask, :3] + blend * estimated_fg[edge_mask]
    # Keep original RGB where solid, and clean zero-alpha pixels
    out[alpha == 0, :3] = 0.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Combine an image with an L-mode alpha mask into RGBA."""
    rgba = image.convert("RGBA")
    rgba.putalpha(mask.convert("L"))
    return rgba


def trim_and_pad(
    image: Image.Image,
    *,
    trim: bool = False,
    pad: int = 0,
    resize: tuple[int, int] | None = None,
) -> Image.Image:
    """Optionally crop to non-transparent bounding box, add uniform padding, and fit-resize."""
    out = image.convert("RGBA")

    if trim:
        bbox = out.getchannel("A").getbbox()
        if bbox is not None:
            out = out.crop(bbox)

    if pad > 0:
        canvas = Image.new("RGBA", (out.width + 2 * pad, out.height + 2 * pad), (0, 0, 0, 0))
        canvas.paste(out, (pad, pad), out)
        out = canvas

    if resize is not None:
        target_w, target_h = resize
        out.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

    return out
