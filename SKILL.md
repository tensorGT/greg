---
name: greg
description: Fast, local, customizable AI background remover and image format converter. Use when asked to remove an image background, make a transparent logo/icon, batch-process images, or convert image formats locally.
---

# `greg` — Agent Usage Guide

`greg` is a local AI background removal and image conversion CLI + MCP server.
It supports **one-line** background removal, folder moving, renaming, format conversion, streaming NDJSON progress, and completion hooks.

## Quick Recipes (CLI)

Always pass `--json -y` when invoking `greg` as an agent:
- `--json` streams one JSON object per line (NDJSON) to `stdout` and flushes immediately.
- `-y` (`--yes`) auto-confirms downloading a model if it is not cached yet, then continues immediately.

### 1. Remove background, move, rename, and convert in one line
```bash
greg remove "path/to/logo.jpg" --out-dir "./assets" --name "brand_logo" --ext webp --preset logo --json -y
```

### 2. Batch process a list of images, glob, or `@filelist.txt`
```bash
greg remove img1.png img2.jpg --out-dir "./clean" --pattern "{stem}_nobg" --ext png --workers 2 --json -y
greg remove @batch.txt --out-dir "./clean" --ext webp --json -y
```

### 3. Sleep / background-wait with completion ping
Run `greg` in the background and either:
- Watch its `stdout` NDJSON stream, **or**
- Pass `--done-file ./job.done.json` (atomically written when finished), **or**
- Pass `--webhook http://localhost:PORT/hook` (HTTP POSTs summary JSON on completion).

```bash
greg remove ./raw/*.png --out-dir ./out --done-file ./out/job.done.json --json -y
```

### 4. Convert format / trim / resize without AI
```bash
greg convert "./assets/brand_logo.png" --out-dir "./public" --name "favicon" --ext ico --trim --json
```

## Presets & Models

| Preset | Model | Best For |
| :--- | :--- | :--- |
| `fast` (default) | `u2netp` (~5 MB) | Ultra-fast general cutouts, icons, bulk previews |
| `logo` | `isnet-general-use` (~178 MB) | Crisp edges, halo defringing, auto-trim + 8px padding |
| `portrait` | `birefnet-general-lite` (~214 MB) | People, hair, fur, soft edges |
| `product` | `isnet-general-use` (~178 MB) | E-commerce products, auto-trim + 16px padding |
| `anime` | `isnet-anime` (~176 MB) | Illustrations, stickers, anime |

Available models (`--model`): `u2netp` (default), `silueta`, `u2net`, `isnet-general-use`, `isnet-anime`, `birefnet-general-lite`, `birefnet-general`.

## Supported Output Formats (`--ext`)
- **With transparency**: `png`, `webp` (`--quality 100` for lossless), `avif`, `tiff`, `tga`, `ico` (multi-size icon)
- **Opaque** (composites onto `--bg-color`, default `white`): `jpg`, `jpeg`, `bmp`, `pdf`

## Useful Flags
- `--out-dir, -o <dir>`: Destination folder (defaults to input file's folder)
- `--name, -n <stem>`: Exact output stem (without extension)
- `--pattern, -p <pat>`: Stem pattern for batches (`{stem}`, `{index}`, `{model}`; default `{stem}_nobg`)
- `--on-exists [suffix|overwrite|skip]`: Collision handling (default `suffix`)
- `--trim / --no-trim`: Crop transparent margins tightly around subject
- `--pad <px>`: Uniform padding around trimmed subject
- `--resize <WxH>`: Fit inside box preserving aspect ratio (e.g. `512x512` or `512`)
- `--bg-color <color>`: `transparent`, `white`, `black`, or `#RRGGBB` / `#RRGGBBAA`
- `--defringe <0.0..1.0>`: Color decontamination on edges (kills white/dark halos)
- `--erode <px>` / `--feather <px>` / `--threshold <0..255>`: Edge tuning
- `--dry-run`: Preview resolved input -> output paths without writing files

## NDJSON Stream Schema (`--json`)
- `{"event": "download_start", "model": "...", "size_mb": 178}` (only if uncached)
- `{"event": "download_progress", "model": "...", "percent": 40, ...}`
- `{"event": "download_done", "model": "..."}`
- `{"event": "start", "action": "remove", "total": 2, "model": "u2netp"}`
- `{"event": "progress", "completed": 1, "total": 2, "input": "...", "output": "...", "status": "ok", "ms": 210}`
- `{"event": "done", "total": 2, "succeeded": 2, "skipped": 0, "failed": 0, "elapsed_ms": 430, "outputs": [...]}`

## Exit Codes
- `0`: All files succeeded
- `1`: Partial failure (check `errors` in final `done` JSON)
- `2`: Invalid arguments or fatal setup error

## Native MCP Server
`greg` can also run as a stdio MCP server (`greg mcp`) exposing:
- `list_models`
- `remove_background`
- `batch_remove_background`
- `convert_image`
