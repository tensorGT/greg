# greg

Fast, local, customizable AI background remover and image pipeline — built for both humans (CLI + Browser UI) and AI agents (NDJSON streaming, completion hooks, MCP server, and `SKILL.md`).

## Features
- **100% Local & Fast**: Powered by ONNX Runtime (`u2netp` default ~5 MB, plus `silueta`, `u2net`, `isnet-general-use`, `isnet-anime`, `birefnet-general-lite`, `birefnet-general`).
- **On-Demand Model Downloads**: Only the tiny default model (`u2netp`, ~5 MB) is needed on first run. Models are stored in your local cache (never committed to Git). Choosing another model prompts to download with a live progress bar and continues immediately without restarting.
- **Pro Edge Tricks**: Built-in color decontamination (halo defringing), morphological erode/dilate, Gaussian feathering, hard thresholding, auto-trim, padding, and aspect-preserving resize.
- **Any Output Format**: Decoupled exporter supports `png`, `webp`, `avif`, `ico` (multi-size), `tiff`, `tga`, `jpg`, `bmp`, and `pdf`.
- **Browser UI (`greg ui`)**: Drag-and-drop batch queue, live download/processing progress, before/after split slider, and ZIP export.
- **AI Agent Ready**: `--json` NDJSON progress stream, `--done-file` / `--webhook` completion pings, `greg agent-docs`, `SKILL.md`, and `greg mcp` stdio MCP server.

## Installation

```bash
git clone <your-repo-url> greg
cd greg
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -e .
```

Optional GPU acceleration (auto-detected by `greg` at runtime):
```bash
# NVIDIA GPU (CUDA):
pip uninstall -y onnxruntime && pip install -e ".[cuda]"

# Windows AMD / Intel / DirectX 12 GPU (DirectML):
pip uninstall -y onnxruntime && pip install -e ".[directml]"
```

Verify your environment:
```bash
greg doctor
```

## Quick Start

### One-line CLI
```bash
# Remove background, move to ./assets, rename to brand_logo, convert to webp
greg remove ./raw/logo.jpg --out-dir ./assets --name brand_logo --ext webp --preset logo

# Batch process a folder
greg remove ./raw_photos --out-dir ./cutouts --ext png --workers 2
```

### Browser UI
```bash
greg ui
```
Opens `http://127.0.0.1:8765` in your default browser.

### Standalone Format Converter
```bash
greg convert ./cutouts/logo.png --out-dir ./dist --ext ico --trim
```

### AI Agent / MCP Usage
See [SKILL.md](SKILL.md) for the full agent reference, or run:
```bash
greg agent-docs
greg mcp
```
