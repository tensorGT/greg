"""Unit and CLI integration tests for greg."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from greg.cli import app
from greg.export.converter import parse_color, save_image
from greg.pipeline import (
    RemoveOptions,
    convert_image_file,
    emit_completion_hooks,
    expand_inputs,
    parse_resize,
    resolve_output_path,
)
from greg.postprocess.refine import apply_mask, defringe_rgba, refine_alpha, trim_and_pad

runner = CliRunner()


def _make_sample_image(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    # Draw a colored square in the center
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (30, 144, 255, 255))
    img.save(path)
    return path


def test_parse_color_and_resize() -> None:
    assert parse_color("transparent") == (0, 0, 0, 0)
    assert parse_color("#ff00aa") == (255, 0, 170, 255)
    assert parse_color("#0f0") == (0, 255, 0, 255)
    assert parse_resize("512x256") == (512, 256)
    assert parse_resize("128") == (128, 128)


def test_alpha_refine_defringe_and_trim() -> None:
    img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    mask = Image.new("L", (64, 64), 0)
    for x in range(20, 44):
        for y in range(20, 44):
            mask.putpixel((x, y), 255)
            img.putpixel((x, y), (20, 100, 200, 255))
    # Semi-transparent halo pixel
    mask.putpixel((19, 20), 128)

    refined = refine_alpha(mask, erode=1, feather=0.5)
    assert refined.size == (64, 64)

    rgba = apply_mask(img, refined)
    defringed = defringe_rgba(rgba, strength=0.8)
    assert defringed.mode == "RGBA"

    trimmed = trim_and_pad(rgba, trim=True, pad=4, resize=(32, 32))
    assert trimmed.width <= 32 and trimmed.height <= 32


def test_save_image_formats(tmp_path: Path) -> None:
    src = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(16, 48):
        for y in range(16, 48):
            src.putpixel((x, y), (200, 50, 50, 255))

    for ext in ("png", "webp", "jpg", "ico"):
        out = save_image(src, tmp_path / f"out.{ext}", ext=ext, bg_color="white" if ext == "jpg" else None)
        assert out.is_file() and out.stat().st_size > 0


def test_expand_inputs_and_resolve_output_path(tmp_path: Path) -> None:
    a = _make_sample_image(tmp_path / "a.png")
    b = _make_sample_image(tmp_path / "b.png")
    filelist = tmp_path / "list.txt"
    filelist.write_text(f"{a}\n{b}\n", encoding="utf-8")

    expanded = expand_inputs([f"@{filelist}"])
    assert expanded == [a.resolve(), b.resolve()]

    opts = RemoveOptions(out_dir=tmp_path / "out", name="logo", ext="webp").resolved()
    p1 = resolve_output_path(a, opts, index=1, total=1)
    assert p1 == (tmp_path / "out" / "logo.webp").resolve()


def test_convert_and_done_file(tmp_path: Path) -> None:
    src = _make_sample_image(tmp_path / "icon.png")
    out_dir = tmp_path / "dist"
    done_file = tmp_path / "job.done.json"

    opts = RemoveOptions(out_dir=out_dir, name="app_icon", ext="ico", trim=True, pad=2).resolved()
    res = convert_image_file(src, opts)
    assert res.status == "ok"
    assert res.output_path is not None and res.output_path.is_file()

    errs = emit_completion_hooks({"status": "ok", "output": str(res.output_path)}, done_file=done_file)
    assert errs == []
    payload = json.loads(done_file.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"


def test_cli_doctor_and_convert(tmp_path: Path) -> None:
    res_doc = runner.invoke(app, ["doctor"])
    assert res_doc.exit_code == 0
    assert "ONNX providers" in res_doc.stdout

    src = _make_sample_image(tmp_path / "sample.png")
    out_dir = tmp_path / "converted"
    res_conv = runner.invoke(
        app,
        [
            "convert",
            str(src),
            "--out-dir",
            str(out_dir),
            "--name",
            "renamed",
            "--ext",
            "webp",
            "--json",
        ],
    )
    assert res_conv.exit_code == 0
    lines = [json.loads(line) for line in res_conv.stdout.strip().splitlines()]
    assert lines[0]["event"] == "start"
    assert lines[-1]["event"] == "done"
    assert lines[-1]["succeeded"] == 1
    assert (out_dir / "renamed.webp").is_file()
