"""Typer CLI for greg — human-friendly and AI-agent-friendly."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from greg import __version__
from greg.engine.models import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    download_model,
    get_model_spec,
    get_models_dir,
    is_model_cached,
)
from greg.engine.session import get_available_providers, get_session
from greg.export.converter import SUPPORTED_FORMATS
from greg.pipeline import (
    OnExistsPolicy,
    ProcessResult,
    RemoveOptions,
    convert_image_file,
    emit_completion_hooks,
    expand_inputs,
    parse_resize,
    process_image_file,
    resolve_output_path,
)
from greg.presets import PRESETS

app = typer.Typer(
    name="greg",
    help="Fast, local, customizable AI background remover & image pipeline.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Inspect and download segmentation models.", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()
err_console = Console(stderr=True)


def _emit_json(payload: dict) -> None:
    """Write one NDJSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _ensure_model_ready(model_name: str, *, auto_yes: bool, json_mode: bool) -> None:
    """Ensure `model_name` is downloaded, prompting if needed, with live progress."""
    if is_model_cached(model_name):
        return

    spec = get_model_spec(model_name)
    is_interactive = sys.stdin.isatty() and not json_mode and not auto_yes

    if is_interactive and model_name != DEFAULT_MODEL:
        confirmed = typer.confirm(
            f"Model '{model_name}' (~{spec.size_mb} MB) is not downloaded yet. Download now and continue?",
            default=True,
        )
        if not confirmed:
            raise typer.Exit(code=2)

    if json_mode:
        _emit_json({"event": "download_start", "model": model_name, "size_mb": spec.size_mb})
        last_pct = -1

        def _on_prog(done: int, total: int) -> None:
            nonlocal last_pct
            pct = int((done / total) * 100) if total > 0 else 0
            if pct >= last_pct + 10 or (total > 0 and done == total):
                last_pct = pct
                _emit_json(
                    {
                        "event": "download_progress",
                        "model": model_name,
                        "downloaded_bytes": done,
                        "total_bytes": total,
                        "percent": pct,
                    }
                )

        download_model(model_name, on_progress=_on_prog)
        _emit_json({"event": "download_done", "model": model_name})
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Downloading {task.fields[model]}[/]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=err_console,
    ) as progress:
        task_id = progress.add_task(
            "download",
            total=spec.size_mb * 1024 * 1024,
            model=model_name,
        )

        def _on_rich_prog(done: int, total: int) -> None:
            if total > 0:
                progress.update(task_id, total=total, completed=done)
            else:
                progress.update(task_id, completed=done)

        download_model(model_name, on_progress=_on_rich_prog)


def _run_batch(
    paths: list[Path],
    opts: RemoveOptions,
    *,
    worker_fn,
    workers: int,
    json_mode: bool,
    dry_run: bool,
    done_file: Path | None,
    webhook: str | None,
    label: str,
) -> int:
    """Execute batch processing with progress reporting and completion hooks."""
    resolved = opts.resolved()
    total = len(paths)

    if dry_run:
        plan = []
        for idx, p in enumerate(paths, start=1):
            out = resolve_output_path(p, resolved, index=idx, total=total)
            plan.append({"input": str(p), "output": str(out)})
        if json_mode:
            _emit_json({"event": "dry_run", "total": total, "items": plan})
        else:
            table = Table(title=f"Dry Run ({total} files)")
            table.add_column("Input", style="cyan")
            table.add_column("Output", style="green")
            for item in plan:
                table.add_row(item["input"], item["output"])
            console.print(table)
        return 0

    t0 = time.perf_counter()
    if json_mode:
        _emit_json({"event": "start", "action": label, "total": total, "model": resolved.model})

    results: list[ProcessResult] = []
    max_workers = max(1, min(workers, total))

    def _run_one(idx_path: tuple[int, Path]) -> ProcessResult:
        idx, p = idx_path
        return worker_fn(p, resolved, index=idx, total=total)

    if json_mode:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, (i, p)): (i, p) for i, p in enumerate(paths, start=1)}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                _emit_json(
                    {
                        "event": "progress",
                        "completed": len(results),
                        "total": total,
                        "input": str(res.input_path),
                        "output": str(res.output_path) if res.output_path else None,
                        "status": res.status,
                        "ms": res.elapsed_ms,
                        "width": res.width,
                        "height": res.height,
                        "error": res.error,
                    }
                )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=err_console,
        ) as progress:
            task_id = progress.add_task(f"{label} ({resolved.model})", total=total)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one, (i, p)): (i, p) for i, p in enumerate(paths, start=1)}
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    progress.advance(task_id, 1)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    succeeded = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "error"]

    summary = {
        "event": "done",
        "action": label,
        "total": total,
        "succeeded": len(succeeded),
        "skipped": len(skipped),
        "failed": len(failed),
        "elapsed_ms": elapsed_ms,
        "outputs": [str(r.output_path) for r in succeeded if r.output_path],
        "errors": [{"input": str(r.input_path), "error": r.error} for r in failed],
    }

    hook_errors = emit_completion_hooks(summary, done_file=done_file, webhook=webhook)
    if hook_errors:
        summary["hook_errors"] = hook_errors

    if json_mode:
        _emit_json(summary)
    else:
        console.print(
            f"[green]Done:[/] {len(succeeded)} ok, {len(skipped)} skipped, "
            f"[red]{len(failed)} failed[/] in {elapsed_ms} ms"
        )
        for r in succeeded[:10]:
            console.print(f"  [dim]->[/] {r.output_path}")
        if len(succeeded) > 10:
            console.print(f"  [dim]...and {len(succeeded) - 10} more[/]")
        for r in failed:
            err_console.print(f"  [red]ERROR[/] {r.input_path}: {r.error}")
        for he in hook_errors:
            err_console.print(f"  [yellow]HOOK WARNING[/] {he}")

    return 1 if failed else 0


@app.command()
def remove(
    inputs: list[str] = typer.Argument(
        ..., help="Image path(s), glob(s), folder(s), or @filelist.txt."
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o", help="Destination folder (defaults to each input's folder)."
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Exact output filename stem (without extension)."
    ),
    pattern: str = typer.Option(
        "{stem}_nobg", "--pattern", "-p", help="Output stem pattern ({stem}, {index}, {model})."
    ),
    ext: str = typer.Option(
        "png", "--ext", "-e", help=f"Output format ({', '.join(SUPPORTED_FORMATS)})."
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Named preset ({', '.join(PRESETS.keys())})."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help=f"Model name (default: {DEFAULT_MODEL})."
    ),
    bg_color: Optional[str] = typer.Option(
        None, "--bg-color", help="Background color (e.g. 'transparent', 'white', '#ff00aa')."
    ),
    quality: int = typer.Option(
        92, "--quality", "-q", min=1, max=100, help="Quality for webp/avif/jpg (100=lossless webp)."
    ),
    trim: Optional[bool] = typer.Option(
        None, "--trim/--no-trim", help="Crop transparent margins tightly around subject."
    ),
    pad: Optional[int] = typer.Option(
        None, "--pad", min=0, help="Uniform padding in pixels after trimming."
    ),
    resize: Optional[str] = typer.Option(
        None, "--resize", help="Fit inside WxH (e.g. 512x512 or 512) keeping aspect ratio."
    ),
    erode: Optional[int] = typer.Option(
        None, "--erode", help="Shrink (>0) or expand (<0) alpha mask in pixels."
    ),
    feather: Optional[float] = typer.Option(
        None, "--feather", help="Gaussian feather radius on alpha edges."
    ),
    threshold: Optional[int] = typer.Option(
        None, "--threshold", min=0, max=255, help="Hard alpha threshold (0..255) for crisp edges."
    ),
    defringe: Optional[float] = typer.Option(
        None, "--defringe", min=0.0, max=1.0, help="Color decontamination strength (0.0..1.0)."
    ),
    on_exists: OnExistsPolicy = typer.Option(
        "suffix", "--on-exists", help="When output exists: suffix, overwrite, or skip."
    ),
    workers: int = typer.Option(
        2, "--workers", "-w", min=1, max=16, help="Parallel worker threads for batch runs."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show resolved input -> output paths without processing."
    ),
    json_mode: bool = typer.Option(
        False, "--json", help="Stream NDJSON progress events to stdout (ideal for AI agents)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Auto-confirm model downloads without prompting."
    ),
    done_file: Optional[Path] = typer.Option(
        None, "--done-file", help="Write atomic JSON summary to this path upon completion."
    ),
    webhook: Optional[str] = typer.Option(
        None, "--webhook", help="POST JSON summary to this URL upon completion."
    ),
) -> None:
    """Remove background from image(s), optionally move, rename, and convert format in one command."""
    try:
        paths = expand_inputs(inputs)
        if not paths:
            raise ValueError("No matching image files found.")
        opts = RemoveOptions(
            model=model,
            preset=preset,
            erode=erode,
            feather=feather,
            threshold=threshold,
            defringe=defringe,
            trim=trim,
            pad=pad,
            resize=parse_resize(resize),
            bg_color=bg_color,
            ext=ext,
            quality=quality,
            out_dir=out_dir,
            name=name,
            pattern=pattern,
            on_exists=on_exists,
        )
        resolved = opts.resolved()
    except Exception as exc:
        if json_mode:
            _emit_json({"event": "error", "error": str(exc)})
        else:
            err_console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=2)

    if not dry_run:
        _ensure_model_ready(resolved.model, auto_yes=yes, json_mode=json_mode)
        # Warm up session once before spawning worker threads
        get_session(resolved.model)

    code = _run_batch(
        paths,
        opts,
        worker_fn=process_image_file,
        workers=workers,
        json_mode=json_mode,
        dry_run=dry_run,
        done_file=done_file,
        webhook=webhook,
        label="remove",
    )
    raise typer.Exit(code=code)


@app.command()
def convert(
    inputs: list[str] = typer.Argument(..., help="Image path(s), glob(s), folder(s), or @filelist.txt."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Destination folder."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Exact output filename stem."),
    pattern: str = typer.Option("{stem}", "--pattern", "-p", help="Output stem pattern."),
    ext: str = typer.Option("webp", "--ext", "-e", help=f"Target format ({', '.join(SUPPORTED_FORMATS)})."),
    bg_color: Optional[str] = typer.Option(None, "--bg-color", help="Background color to composite onto."),
    quality: int = typer.Option(92, "--quality", "-q", min=1, max=100, help="Output quality."),
    trim: bool = typer.Option(False, "--trim/--no-trim", help="Crop transparent margins."),
    pad: int = typer.Option(0, "--pad", min=0, help="Uniform padding in pixels."),
    resize: Optional[str] = typer.Option(None, "--resize", help="Fit inside WxH keeping aspect ratio."),
    on_exists: OnExistsPolicy = typer.Option("suffix", "--on-exists", help="suffix, overwrite, or skip."),
    workers: int = typer.Option(4, "--workers", "-w", min=1, max=16, help="Parallel worker threads."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview resolved paths."),
    json_mode: bool = typer.Option(False, "--json", help="Stream NDJSON progress events."),
    done_file: Optional[Path] = typer.Option(None, "--done-file", help="Write completion JSON file."),
    webhook: Optional[str] = typer.Option(None, "--webhook", help="POST completion JSON to URL."),
) -> None:
    """Convert image format, trim, pad, or resize without running AI background removal."""
    try:
        paths = expand_inputs(inputs)
        if not paths:
            raise ValueError("No matching image files found.")
        opts = RemoveOptions(
            trim=trim,
            pad=pad,
            resize=parse_resize(resize),
            bg_color=bg_color,
            ext=ext,
            quality=quality,
            out_dir=out_dir,
            name=name,
            pattern=pattern,
            on_exists=on_exists,
        )
    except Exception as exc:
        if json_mode:
            _emit_json({"event": "error", "error": str(exc)})
        else:
            err_console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=2)

    code = _run_batch(
        paths,
        opts,
        worker_fn=convert_image_file,
        workers=workers,
        json_mode=json_mode,
        dry_run=dry_run,
        done_file=done_file,
        webhook=webhook,
        label="convert",
    )
    raise typer.Exit(code=code)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", help="Port number."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically."),
) -> None:
    """Launch the local browser UI for interactive background removal and batch jobs."""
    from greg.ui.server import launch_ui

    launch_ui(host=host, port=port, open_browser=open_browser)


@app.command()
def mcp() -> None:
    """Run greg as a stdio MCP server for AI agents."""
    from greg.mcp_server import run_mcp_server

    run_mcp_server()


@models_app.command("list")
def models_list() -> None:
    """List available segmentation models and their local cache status."""
    table = Table(title=f"greg Models (cache: {get_models_dir()})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Description")

    for spec in MODEL_REGISTRY.values():
        cached = is_model_cached(spec.name)
        status = "[green]cached[/]" if cached else "[dim]not downloaded[/]"
        name_label = f"{spec.name} (default)" if spec.name == DEFAULT_MODEL else spec.name
        table.add_row(name_label, f"~{spec.size_mb} MB", status, spec.description)
    console.print(table)


@models_app.command("download")
def models_download(
    model_name: str = typer.Argument(..., help="Model name to download."),
) -> None:
    """Download a segmentation model into the local cache."""
    get_model_spec(model_name)
    _ensure_model_ready(model_name, auto_yes=True, json_mode=False)
    console.print(f"[green]Ready:[/] {model_name}")


@app.command()
def doctor() -> None:
    """Check Python environment, ONNX Runtime providers, cached models, and supported formats."""
    console.print(f"[bold]greg[/] v{__version__}")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"ONNX providers: {', '.join(get_available_providers())}")
    console.print(f"Models dir: {get_models_dir()}")
    cached = [m for m in MODEL_REGISTRY if is_model_cached(m)]
    console.print(f"Cached models: {', '.join(cached) if cached else 'none (default u2netp downloads on first run)'}")
    console.print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")


@app.command("agent-docs")
def agent_docs() -> None:
    """Print concise machine-readable usage instructions for AI agents."""
    docs = """# greg CLI Quick Reference for AI Agents
- One-line remove + move + rename + convert:
  greg remove <input> --out-dir <dir> --name <stem> --ext <png|webp|ico|jpg|...> --preset <fast|logo|portrait|product|anime> --json -y
- Batch list:
  greg remove img1.png img2.jpg --out-dir ./out --pattern "{stem}_nobg" --ext webp --json -y
- Convert format only (no AI):
  greg convert input.png --out-dir ./dist --ext ico --trim --json
- NDJSON stdout events:
  start -> download_start/download_progress/download_done (if uncached) -> progress (per file) -> done
- Completion hooks:
  --done-file <path>  (writes atomic JSON summary on finish)
  --webhook <url>     (POSTs JSON summary on finish)
- Exit codes: 0=all ok, 1=partial failure, 2=fatal/bad args
"""
    sys.stdout.write(docs)


def main() -> None:
    """Entrypoint for `greg` console script."""
    app()


if __name__ == "__main__":
    main()
