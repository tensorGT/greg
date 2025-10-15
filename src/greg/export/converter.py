"""Decoupled image format converter and exporter supporting alpha and non-alpha formats."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ALPHA_FORMATS = {"png", "webp", "avif", "tiff", "tif", "tga", "ico"}
OPAQUE_FORMATS = {"jpg", "jpeg", "bmp", "pdf"}
SUPPORTED_FORMATS = sorted(ALPHA_FORMATS | OPAQUE_FORMATS)

NAMED_COLORS: dict[str, tuple[int, int, int, int]] = {
    "transparent": (0, 0, 0, 0),
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "red": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "blue": (0, 0, 255, 255),
}


def parse_color(color_str: str | None) -> tuple[int, int, int, int]:
    """Parse a color name or hex string (#RGB, #RRGGBB, #RRGGBBAA) into RGBA tuple."""
    if not color_str:
        return (0, 0, 0, 0)
    key = color_str.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    if key.startswith("#"):
        hex_part = key[1:]
        if len(hex_part) == 3:
            r, g, b = (int(c * 2, 16) for c in hex_part)
            return (r, g, b, 255)
        if len(hex_part) == 6:
            r = int(hex_part[0:2], 16)
            g = int(hex_part[2:4], 16)
            b = int(hex_part[4:6], 16)
            return (r, g, b, 255)
        if len(hex_part) == 8:
            r = int(hex_part[0:2], 16)
            g = int(hex_part[2:4], 16)
            b = int(hex_part[4:6], 16)
            a = int(hex_part[6:8], 16)
            return (r, g, b, a)
    raise ValueError(
        f"Invalid color '{color_str}'. Use 'transparent', 'white', 'black', or hex like '#ffffff'."
    )


def apply_background(image: Image.Image, bg_color: str | None) -> Image.Image:
    """Composite an RGBA image onto `bg_color` if provided and not transparent."""
    rgba = image.convert("RGBA")
    color = parse_color(bg_color)
    if color[3] == 0:
        return rgba
    bg = Image.new("RGBA", rgba.size, color)
    return Image.alpha_composite(bg, rgba)


def save_image(
    image: Image.Image,
    out_path: Path,
    *,
    ext: str | None = None,
    quality: int = 92,
    bg_color: str | None = None,
) -> Path:
    """Save `image` to `out_path` in the requested format, handling alpha vs opaque formats."""
    fmt = (ext or out_path.suffix.lstrip(".") or "png").lower()
    if fmt not in ALPHA_FORMATS and fmt not in OPAQUE_FORMATS:
        raise ValueError(
            f"Unsupported output format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = apply_background(image, bg_color)

    if fmt in OPAQUE_FORMATS:
        # Composite onto white if still transparent before dropping alpha
        if prepared.mode == "RGBA":
            white_bg = Image.new("RGBA", prepared.size, (255, 255, 255, 255))
            prepared = Image.alpha_composite(white_bg, prepared).convert("RGB")
        else:
            prepared = prepared.convert("RGB")

        if fmt in {"jpg", "jpeg"}:
            prepared.save(out_path, format="JPEG", quality=quality, optimize=True)
        elif fmt == "bmp":
            prepared.save(out_path, format="BMP")
        elif fmt == "pdf":
            prepared.save(out_path, format="PDF", resolution=150.0)
        return out_path

    # Alpha-capable formats
    if fmt == "png":
        prepared.save(out_path, format="PNG", optimize=True)
    elif fmt == "webp":
        lossless = quality >= 100
        prepared.save(out_path, format="WEBP", quality=quality, lossless=lossless, method=4)
    elif fmt == "avif":
        prepared.save(out_path, format="AVIF", quality=quality)
    elif fmt in {"tiff", "tif"}:
        prepared.save(out_path, format="TIFF", compression="tiff_deflate")
    elif fmt == "tga":
        prepared.save(out_path, format="TGA")
    elif fmt == "ico":
        sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        prepared.save(out_path, format="ICO", sizes=sizes)

    return out_path
