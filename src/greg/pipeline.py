"""Core image processing pipeline, input expansion, path resolution, and completion hooks."""

from __future__ import annotations

import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from PIL import Image, ImageOps

from greg.engine.models import DEFAULT_MODEL
from greg.engine.session import segment_image
from greg.export.converter import save_image
from greg.postprocess.refine import apply_mask, defringe_rgba, refine_alpha, trim_and_pad
from greg.presets import get_preset

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".avif",
    ".bmp",
    ".tif",
    ".tiff",
    ".tga",
    ".ico",
}

OnExistsPolicy = Literal["suffix", "overwrite", "skip"]


@dataclass
class RemoveOptions:
    """Full configuration for a background-removal job."""

    model: str | None = None
    preset: str | None = None
    erode: int | None = None
    feather: float | None = None
    threshold: int | None = None
    defringe: float | None = None
    trim: bool | None = None
    pad: int | None = None
    resize: tuple[int, int] | None = None
    bg_color: str | None = None
    ext: str = "png"
    quality: int = 92
    out_dir: Path | None = None
    name: str | None = None
    pattern: str = "{stem}_nobg"
    on_exists: OnExistsPolicy = "suffix"

    def resolved(self) -> "ResolvedRemoveOptions":
        """Merge preset defaults with explicit overrides."""
        base = get_preset(self.preset) if self.preset else None
        return ResolvedRemoveOptions(
            model=self.model or (base.model if base else DEFAULT_MODEL),
            erode=self.erode if self.erode is not None else (base.erode if base else 0),
            feather=self.feather if self.feather is not None else (base.feather if base else 0.0),
            threshold=self.threshold if self.threshold is not None else (base.threshold if base else None),
            defringe=self.defringe if self.defringe is not None else (base.defringe if base else 0.7),
            trim=self.trim if self.trim is not None else (base.trim if base else False),
            pad=self.pad if self.pad is not None else (base.pad if base else 0),
            resize=self.resize,
            bg_color=self.bg_color,
            ext=self.ext.lstrip(".").lower() or "png",
            quality=self.quality,
            out_dir=self.out_dir,
            name=self.name,
            pattern=self.pattern,
            on_exists=self.on_exists,
        )


@dataclass
class ResolvedRemoveOptions:
    """Concrete, non-None options ready for execution."""

    model: str
    erode: int
    feather: float
    threshold: int | None
    defringe: float
    trim: bool
    pad: int
    resize: tuple[int, int] | None
    bg_color: str | None
    ext: str
    quality: int
    out_dir: Path | None
    name: str | None
    pattern: str
    on_exists: OnExistsPolicy


@dataclass
class ProcessResult:
    """Outcome for a single processed image."""

    input_path: Path
    output_path: Path | None
    status: Literal["ok", "skipped", "error"]
    elapsed_ms: int
    width: int | None = None
    height: int | None = None
    error: str | None = None


def parse_resize(value: str | None) -> tuple[int, int] | None:
    """Parse 'WxH' or 'N' (square) into (width, height)."""
    if not value:
        return None
    s = value.strip().lower()
    if "x" in s:
        parts = s.split("x", 1)
        w, h = int(parts[0]), int(parts[1])
    else:
        w = h = int(s)
    if w <= 0 or h <= 0:
        raise ValueError(f"Resize dimensions must be positive: '{value}'")
    return (w, h)


def expand_inputs(raw_inputs: list[str]) -> list[Path]:
    """Expand file paths, globs, directories, and @filelist.txt entries into deduplicated image Paths."""
    collected: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        resolved = p.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            collected.append(resolved)

    for item in raw_inputs:
        if item.startswith("@"):
            list_path = Path(item[1:]).expanduser()
            if not list_path.is_file():
                raise FileNotFoundError(f"Input list file not found: {list_path}")
            for line in list_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    _add(Path(line).expanduser())
            continue

        p = Path(item).expanduser()
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    _add(child)
        elif any(ch in item for ch in ("*", "?", "[")):
            for match in sorted(glob.glob(item, recursive=True)):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() in IMAGE_EXTENSIONS:
                    _add(mp)
        else:
            if not p.exists():
                raise FileNotFoundError(f"Input file not found: {p}")
            _add(p)

    return collected


def resolve_output_path(
    input_path: Path,
    opts: ResolvedRemoveOptions,
    *,
    index: int,
    total: int,
) -> Path:
    """Compute target output Path given out_dir, name, pattern, ext, and on_exists policy."""
    target_dir = opts.out_dir.resolve() if opts.out_dir else input_path.parent
    ext = opts.ext.lstrip(".")

    if opts.name:
        stem = opts.name if total == 1 else f"{opts.name}_{index}"
    else:
        stem = opts.pattern.format(
            stem=input_path.stem,
            index=index,
            model=opts.model,
        )

    candidate = target_dir / f"{stem}.{ext}"
    if opts.on_exists == "overwrite" or opts.on_exists == "skip" or not candidate.exists():
        return candidate

    counter = 1
    while True:
        alt = target_dir / f"{stem}_{counter}.{ext}"
        if not alt.exists():
            return alt
        counter += 1


def process_image_pil(image: Image.Image, opts: ResolvedRemoveOptions) -> Image.Image:
    """Run segmentation + alpha refinement + defringing + trim/pad/resize on an in-memory PIL image."""
    oriented = ImageOps.exif_transpose(image)
    raw_mask = segment_image(oriented, opts.model)
    mask = refine_alpha(
        raw_mask,
        erode=opts.erode,
        feather=opts.feather,
        threshold=opts.threshold,
    )
    rgba = apply_mask(oriented, mask)
    if opts.defringe > 0.0:
        rgba = defringe_rgba(rgba, strength=opts.defringe)
    if opts.trim or opts.pad > 0 or opts.resize is not None:
        rgba = trim_and_pad(rgba, trim=opts.trim, pad=opts.pad, resize=opts.resize)
    return rgba


def process_image_file(
    input_path: Path,
    opts: ResolvedRemoveOptions,
    *,
    index: int = 1,
    total: int = 1,
) -> ProcessResult:
    """Process a single image file from disk and write output according to `opts`."""
    t0 = time.perf_counter()
    out_path = resolve_output_path(input_path, opts, index=index, total=total)

    if opts.on_exists == "skip" and out_path.exists():
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ProcessResult(
            input_path=input_path,
            output_path=out_path,
            status="skipped",
            elapsed_ms=elapsed,
        )

    try:
        with Image.open(input_path) as img:
            result_img = process_image_pil(img, opts)
        save_image(
            result_img,
            out_path,
            ext=opts.ext,
            quality=opts.quality,
            bg_color=opts.bg_color,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ProcessResult(
            input_path=input_path,
            output_path=out_path,
            status="ok",
            elapsed_ms=elapsed,
            width=result_img.width,
            height=result_img.height,
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ProcessResult(
            input_path=input_path,
            output_path=None,
            status="error",
            elapsed_ms=elapsed,
            error=str(exc),
        )


def convert_image_file(
    input_path: Path,
    opts: ResolvedRemoveOptions,
    *,
    index: int = 1,
    total: int = 1,
) -> ProcessResult:
    """Convert/trim/pad/resize an existing image without running AI background removal."""
    t0 = time.perf_counter()
    out_path = resolve_output_path(input_path, opts, index=index, total=total)

    if opts.on_exists == "skip" and out_path.exists():
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ProcessResult(
            input_path=input_path,
            output_path=out_path,
            status="skipped",
            elapsed_ms=elapsed,
        )

    try:
        with Image.open(input_path) as img:
            out_img = ImageOps.exif_transpose(img).convert("RGBA")
        if opts.trim or opts.pad > 0 or opts.resize is not None:
            out_img = trim_and_pad(out_img, trim=opts.trim, pad=opts.pad, resize=opts.resize)
        save_image(
            out_img,
            out_path,
            ext=opts.ext,
            quality=opts.quality,
            bg_color=opts.bg_color,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ProcessResult(
            input_path=input_path,
            output_path=out_path,
            status="ok",
            elapsed_ms=elapsed,
            width=out_img.width,
            height=out_img.height,
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return ProcessResult(
            input_path=input_path,
            output_path=None,
            status="error",
            elapsed_ms=elapsed,
            error=str(exc),
        )


def emit_completion_hooks(
    summary: dict,
    *,
    done_file: Path | None = None,
    webhook: str | None = None,
) -> list[str]:
    """Write atomic done_file and/or POST summary JSON to webhook URL. Returns list of hook errors."""
    errors: list[str] = []
    if done_file is not None:
        try:
            done_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = done_file.with_suffix(done_file.suffix + ".tmp")
            tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            tmp.replace(done_file)
        except Exception as exc:
            errors.append(f"done-file error: {exc}")

    if webhook:
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(webhook, json=summary)
                resp.raise_for_status()
        except Exception as exc:
            errors.append(f"webhook error: {exc}")

    return errors
