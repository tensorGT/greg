"""Stdio MCP server exposing greg background-removal and conversion tools to AI agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from greg.engine.models import DEFAULT_MODEL, MODEL_REGISTRY, download_model, is_model_cached
from greg.engine.session import get_session
from greg.pipeline import (
    RemoveOptions,
    convert_image_file,
    expand_inputs,
    parse_resize,
    process_image_file,
)

mcp_app = FastMCP("greg")


@mcp_app.tool()
def list_models() -> list[dict[str, Any]]:
    """List available background-removal models, their sizes, and whether they are cached locally."""
    return [
        {
            "name": m.name,
            "default": m.name == DEFAULT_MODEL,
            "size_mb": m.size_mb,
            "cached": is_model_cached(m.name),
            "description": m.description,
        }
        for m in MODEL_REGISTRY.values()
    ]


@mcp_app.tool()
def remove_background(
    input_path: str,
    out_dir: str | None = None,
    name: str | None = None,
    ext: str = "png",
    preset: str | None = None,
    model: str | None = None,
    bg_color: str | None = None,
    trim: bool | None = None,
    pad: int | None = None,
    resize: str | None = None,
    defringe: float | None = None,
    erode: int | None = None,
    feather: float | None = None,
    on_exists: str = "suffix",
) -> dict[str, Any]:
    """Remove background from a single image, optionally moving, renaming, and converting format."""
    opts = RemoveOptions(
        model=model,
        preset=preset,
        erode=erode,
        feather=feather,
        defringe=defringe,
        trim=trim,
        pad=pad,
        resize=parse_resize(resize),
        bg_color=bg_color,
        ext=ext,
        out_dir=Path(out_dir) if out_dir else None,
        name=name,
        on_exists=on_exists,  # type: ignore[arg-type]
    )
    resolved = opts.resolved()
    if not is_model_cached(resolved.model):
        download_model(resolved.model)
    get_session(resolved.model)

    res = process_image_file(Path(input_path).expanduser().resolve(), resolved)
    return {
        "status": res.status,
        "input": str(res.input_path),
        "output": str(res.output_path) if res.output_path else None,
        "width": res.width,
        "height": res.height,
        "elapsed_ms": res.elapsed_ms,
        "error": res.error,
    }


@mcp_app.tool()
def batch_remove_background(
    inputs: list[str],
    out_dir: str | None = None,
    pattern: str = "{stem}_nobg",
    ext: str = "png",
    preset: str | None = None,
    model: str | None = None,
    trim: bool | None = None,
    pad: int | None = None,
    on_exists: str = "suffix",
) -> dict[str, Any]:
    """Remove background from multiple images, globs, or folders in one call."""
    paths = expand_inputs(inputs)
    opts = RemoveOptions(
        model=model,
        preset=preset,
        trim=trim,
        pad=pad,
        ext=ext,
        out_dir=Path(out_dir) if out_dir else None,
        pattern=pattern,
        on_exists=on_exists,  # type: ignore[arg-type]
    )
    resolved = opts.resolved()
    if not is_model_cached(resolved.model):
        download_model(resolved.model)
    get_session(resolved.model)

    results = [
        process_image_file(p, resolved, index=i, total=len(paths))
        for i, p in enumerate(paths, start=1)
    ]
    return {
        "total": len(results),
        "succeeded": sum(1 for r in results if r.status == "ok"),
        "failed": sum(1 for r in results if r.status == "error"),
        "outputs": [str(r.output_path) for r in results if r.output_path],
        "errors": [{"input": str(r.input_path), "error": r.error} for r in results if r.error],
    }


@mcp_app.tool()
def convert_image(
    input_path: str,
    ext: str = "webp",
    out_dir: str | None = None,
    name: str | None = None,
    bg_color: str | None = None,
    quality: int = 92,
    trim: bool = False,
    pad: int = 0,
    resize: str | None = None,
) -> dict[str, Any]:
    """Convert image format, trim, pad, or resize without running AI background removal."""
    opts = RemoveOptions(
        trim=trim,
        pad=pad,
        resize=parse_resize(resize),
        bg_color=bg_color,
        ext=ext,
        quality=quality,
        out_dir=Path(out_dir) if out_dir else None,
        name=name,
        pattern="{stem}",
    )
    resolved = opts.resolved()
    res = convert_image_file(Path(input_path).expanduser().resolve(), resolved)
    return {
        "status": res.status,
        "input": str(res.input_path),
        "output": str(res.output_path) if res.output_path else None,
        "width": res.width,
        "height": res.height,
        "elapsed_ms": res.elapsed_ms,
        "error": res.error,
    }


def run_mcp_server() -> None:
    """Start the greg MCP server over stdio."""
    mcp_app.run()
