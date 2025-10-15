"""FastAPI server + SSE job queue for the greg browser UI."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import uuid
import webbrowser
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from PIL import Image, ImageOps

from greg.engine.models import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    download_model,
    get_model_spec,
    is_model_cached,
)
from greg.engine.session import get_session
from greg.export.converter import SUPPORTED_FORMATS, save_image
from greg.pipeline import RemoveOptions, process_image_pil
from greg.presets import PRESETS

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class JobOutput:
    """Single processed output stored in memory for preview and download."""

    filename: str
    content_type: str
    data: bytes
    width: int
    height: int
    ms: int


@dataclass
class UIJob:
    """In-memory state for an async UI background-removal job."""

    job_id: str
    total: int
    events: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[JobOutput] = field(default_factory=list)
    done: bool = False
    condition: threading.Condition = field(default_factory=threading.Condition)

    def push_event(self, event: dict[str, Any]) -> None:
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()


_JOBS: dict[str, UIJob] = {}
_JOBS_LOCK = threading.Lock()

CONTENT_TYPES = {
    "png": "image/png",
    "webp": "image/webp",
    "avif": "image/avif",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "ico": "image/x-icon",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "tga": "image/x-tga",
    "pdf": "application/pdf",
}


def _run_ui_job(
    job: UIJob,
    files: list[tuple[str, bytes]],
    opts: RemoveOptions,
) -> None:
    """Background worker that downloads model if needed, then processes all uploaded images."""
    import time

    try:
        resolved = opts.resolved()
        if not is_model_cached(resolved.model):
            spec = get_model_spec(resolved.model)
            job.push_event(
                {"event": "download_start", "model": resolved.model, "size_mb": spec.size_mb}
            )
            last_pct = -1

            def _on_prog(done: int, total: int) -> None:
                nonlocal last_pct
                pct = int((done / total) * 100) if total > 0 else 0
                if pct >= last_pct + 5 or (total > 0 and done == total):
                    last_pct = pct
                    job.push_event(
                        {
                            "event": "download_progress",
                            "model": resolved.model,
                            "percent": pct,
                            "downloaded_bytes": done,
                            "total_bytes": total,
                        }
                    )

            download_model(resolved.model, on_progress=_on_prog)
            job.push_event({"event": "download_done", "model": resolved.model})

        get_session(resolved.model)
        job.push_event({"event": "start", "total": len(files), "model": resolved.model})

        for idx, (orig_name, raw_bytes) in enumerate(files):
            t0 = time.perf_counter()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                oriented = ImageOps.exif_transpose(img)
                result_img = process_image_pil(oriented, resolved)

            buf = io.BytesIO()
            tmp_path = Path(f"out.{resolved.ext}")
            # Save into BytesIO via format logic
            _save_to_bytesio(
                result_img,
                buf,
                ext=resolved.ext,
                quality=resolved.quality,
                bg_color=resolved.bg_color,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            stem = Path(orig_name).stem
            out_name = f"{stem}_nobg.{resolved.ext}"
            out_item = JobOutput(
                filename=out_name,
                content_type=CONTENT_TYPES.get(resolved.ext, "application/octet-stream"),
                data=buf.getvalue(),
                width=result_img.width,
                height=result_img.height,
                ms=elapsed_ms,
            )
            job.outputs.append(out_item)
            job.push_event(
                {
                    "event": "progress",
                    "index": idx,
                    "completed": idx + 1,
                    "total": len(files),
                    "filename": out_name,
                    "width": result_img.width,
                    "height": result_img.height,
                    "ms": elapsed_ms,
                    "url": f"/api/jobs/{job.job_id}/file/{idx}",
                }
            )

        job.push_event(
            {
                "event": "done",
                "total": len(files),
                "zip_url": f"/api/jobs/{job.job_id}/zip" if len(files) > 1 else None,
            }
        )
    except Exception as exc:
        job.push_event({"event": "error", "error": str(exc)})
    finally:
        with job.condition:
            job.done = True
            job.condition.notify_all()


def _save_to_bytesio(
    image: Image.Image,
    buf: io.BytesIO,
    *,
    ext: str,
    quality: int,
    bg_color: str | None,
) -> None:
    """Save PIL image into an in-memory BytesIO using the shared converter rules."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / f"result.{ext}"
        save_image(image, out_path, ext=ext, quality=quality, bg_color=bg_color)
        buf.write(out_path.read_bytes())


def create_app() -> FastAPI:
    """Create the FastAPI application for the greg browser UI."""
    app = FastAPI(title="greg UI", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "default_model": DEFAULT_MODEL,
            "models": [
                {
                    "name": m.name,
                    "size_mb": m.size_mb,
                    "cached": is_model_cached(m.name),
                    "description": m.description,
                }
                for m in MODEL_REGISTRY.values()
            ],
            "presets": [
                {"name": p.name, "model": p.model, "description": p.description}
                for p in PRESETS.values()
            ],
            "formats": SUPPORTED_FORMATS,
        }

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile] = File(...),
        model: str = Form(DEFAULT_MODEL),
        preset: str = Form(""),
        ext: str = Form("png"),
        bg_color: str = Form("transparent"),
        quality: int = Form(92),
        defringe: float = Form(0.7),
        erode: int = Form(0),
        feather: float = Form(0.0),
        trim: bool = Form(False),
        pad: int = Form(0),
    ) -> dict[str, str]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        loaded: list[tuple[str, bytes]] = []
        for f in files:
            loaded.append((f.filename or "image.png", await f.read()))

        opts = RemoveOptions(
            model=model or None,
            preset=preset or None,
            ext=ext,
            bg_color=None if bg_color == "transparent" else bg_color,
            quality=quality,
            defringe=defringe,
            erode=erode,
            feather=feather,
            trim=trim,
            pad=pad,
        )
        job_id = uuid.uuid4().hex[:12]
        job = UIJob(job_id=job_id, total=len(loaded))
        with _JOBS_LOCK:
            _JOBS[job_id] = job

        thread = threading.Thread(target=_run_ui_job, args=(job, loaded, opts), daemon=True)
        thread.start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        async def _stream():
            cursor = 0
            while True:
                with job.condition:
                    while cursor >= len(job.events) and not job.done:
                        job.condition.wait(timeout=0.25)
                    batch = job.events[cursor:]
                    cursor = len(job.events)
                    is_done = job.done

                for ev in batch:
                    yield f"data: {json.dumps(ev)}\n\n"

                if is_done and cursor >= len(job.events):
                    break
                await asyncio.sleep(0.02)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/file/{index}")
    def get_job_file(job_id: str, index: int) -> Response:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None or index < 0 or index >= len(job.outputs):
            raise HTTPException(status_code=404, detail="Output not found.")
        item = job.outputs[index]
        return Response(
            content=item.data,
            media_type=item.content_type,
            headers={"Content-Disposition": f'inline; filename="{item.filename}"'},
        )

    @app.get("/api/jobs/{job_id}/zip")
    def get_job_zip(job_id: str) -> Response:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None or not job.outputs:
            raise HTTPException(status_code=404, detail="Job outputs not found.")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in job.outputs:
                zf.writestr(item.filename, item.data)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="greg_{job_id}.zip"'},
        )

    return app


def launch_ui(*, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the Uvicorn server and optionally open the browser."""
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
