import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.embedder import DinoV2Embedder
from app.images import InvalidImageError, decode_image, query_views
from app.store import IndexItem, create_vector_store

logger = logging.getLogger("visual-search")
logging.basicConfig(level=logging.INFO)


class Runtime:
    embedder: DinoV2Embedder | None = None
    store: object | None = None


runtime = Runtime()
scan_jobs: dict[str, dict[str, object]] = {}
scan_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logger.info("Loading visual model %s (%s)", settings.model_id, settings.model_version)
    runtime.embedder = await run_in_threadpool(DinoV2Embedder, settings)
    runtime.store = await run_in_threadpool(create_vector_store, settings, runtime.embedder.dimension)
    logger.info("Visual search ready: device=%s dimension=%d collection=%s",
                runtime.embedder.device, runtime.embedder.dimension, settings.collection_name)
    yield


app = FastAPI(title="Product Visual Search", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/admin")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


class EvidenceResponse(BaseModel):
    file_id: str
    image_url: str | None
    score: float


class ProductMatchResponse(BaseModel):
    product_id: str
    product_code: str
    score: float
    confidence: str
    evidence: list[EvidenceResponse]


class SearchResponse(BaseModel):
    request_id: str
    model_version: str
    elapsed_ms: int
    matches: list[ProductMatchResponse]


class IndexResponse(BaseModel):
    point_id: str
    model_version: str
    indexed: bool = True


class BatchIndexMetadata(BaseModel):
    product_id: str
    product_code: str
    file_id: str
    category: str | None = None
    image_url: str | None = None


class BatchIndexResponse(BaseModel):
    point_ids: list[str]
    model_version: str


class CatalogScanRequest(BaseModel):
    prefix: str = ""
    remove_missing: bool = True


class CatalogScanResponse(BaseModel):
    job_id: str
    status: str
    total: int
    indexed: int
    skipped: int = 0
    failed: int
    removed: int = 0
    current: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    prefix: str = ""
    remove_missing: bool = True
    created_at: str | None = None


class CatalogStatsResponse(BaseModel):
    backend: str
    location: str
    collection: str
    model_version: str
    indexed_count: int
    source_bytes: int | None = None
    database_bytes: int | None = None
    last_updated_at: str | None = None
    vector_dimension: int | None = None
    active_job: CatalogScanResponse | None = None
    latest_job: CatalogScanResponse | None = None


class CatalogAssetsResponse(BaseModel):
    offset: int
    limit: int
    total: int
    items: list[dict]


async def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的 API Key")


async def read_upload(image: UploadFile, settings: Settings) -> bytes:
    data = await image.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="图片超过大小限制")
    return data


def confidence(score: float, settings: Settings) -> str:
    if score >= settings.high_confidence_score:
        return "high"
    if score >= settings.medium_confidence_score:
        return "medium"
    return "low"


def configured_catalog_roots(settings: Settings) -> list[Path]:
    raw = settings.catalog_roots or settings.catalog_root
    parts = [item.strip() for item in raw.replace("\n", ";").split(";") if item.strip()]
    roots: list[Path] = []
    for item in parts:
        path = Path(item).resolve()
        if path not in roots:
            roots.append(path)
    return roots


@app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search(
    image: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form(ge=1, le=50)] = 10,
    category: Annotated[str | None, Form()] = None,
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    started = time.perf_counter()
    try:
        decoded = decode_image(await read_upload(image, settings), settings.max_image_pixels)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    vectors = await run_in_threadpool(runtime.embedder.embed, query_views(decoded))
    matches = await run_in_threadpool(runtime.store.search, vectors, top_k, category)
    return SearchResponse(
        request_id=str(uuid4()),
        model_version=settings.model_version,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        matches=[ProductMatchResponse(
            product_id=result.product_id,
            product_code=result.product_code,
            score=result.score,
            confidence=confidence(result.score, settings),
            evidence=[EvidenceResponse(**item.__dict__) for item in result.evidence],
        ) for result in matches],
    )


@app.post("/v1/index/image", response_model=IndexResponse,
          dependencies=[Depends(require_api_key)])
async def index_image(
    image: Annotated[UploadFile, File()],
    product_id: Annotated[str, Form(min_length=1, max_length=128)],
    product_code: Annotated[str, Form(min_length=1, max_length=128)],
    file_id: Annotated[str, Form(min_length=1, max_length=128)],
    category: Annotated[str | None, Form(max_length=128)] = None,
    image_url: Annotated[str | None, Form(max_length=2048)] = None,
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    data = await read_upload(image, settings)
    signature = hashlib.sha256(data).hexdigest()
    existing_signature = await run_in_threadpool(
        runtime.store.get_signature, file_id, settings.model_version)
    if existing_signature == signature:
        point_id = str(uuid5(NAMESPACE_URL, f"{settings.model_version}:{file_id}"))
        return IndexResponse(
            point_id=point_id, model_version=settings.model_version, indexed=False)
    try:
        decoded = decode_image(data, settings.max_image_pixels)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    vector = (await run_in_threadpool(runtime.embedder.embed, [decoded]))[0]
    point_id = await run_in_threadpool(
        runtime.store.upsert_image,
        product_id=product_id,
        product_code=product_code,
        file_id=file_id,
        category=category,
        image_url=image_url,
        vector=vector,
        model_version=settings.model_version,
        signature=signature,
        file_size=len(data),
    )
    return IndexResponse(point_id=point_id, model_version=settings.model_version)


@app.post("/v1/index/batch", response_model=BatchIndexResponse,
          dependencies=[Depends(require_api_key)])
async def index_batch(
    images: Annotated[list[UploadFile], File()],
    metadata: Annotated[str, Form()],
    settings: Settings = Depends(get_settings),
) -> BatchIndexResponse:
    if not images or len(images) > settings.max_batch_size:
        raise HTTPException(status_code=400,
                            detail=f"批次图片数必须在 1-{settings.max_batch_size} 之间")
    try:
        items = [BatchIndexMetadata.model_validate(item) for item in json.loads(metadata)]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="metadata 不是有效的 JSON 数组") from exc
    if len(items) != len(images):
        raise HTTPException(status_code=422, detail="metadata 数量必须与图片数量相同")

    decoded_images = []
    image_data: list[bytes] = []
    total_bytes = 0
    for upload in images:
        data = await read_upload(upload, settings)
        total_bytes += len(data)
        if total_bytes > settings.max_batch_bytes:
            raise HTTPException(status_code=413, detail="批次总大小超过限制")
        try:
            decoded_images.append(decode_image(data, settings.max_image_pixels))
            image_data.append(data)
        except InvalidImageError as exc:
            raise HTTPException(status_code=422,
                                detail=f"{upload.filename}: {exc}") from exc

    vectors = await run_in_threadpool(runtime.embedder.embed, decoded_images)
    index_items = [IndexItem(
        product_id=item.product_id,
        product_code=item.product_code,
        file_id=item.file_id,
        category=item.category,
        image_url=item.image_url,
        vector=vector,
        signature=hashlib.sha256(data).hexdigest(),
        file_size=len(data),
    ) for item, vector, data in zip(items, vectors, image_data)]
    point_ids = await run_in_threadpool(
        runtime.store.upsert_images, index_items, settings.model_version)
    return BatchIndexResponse(point_ids=point_ids, model_version=settings.model_version)


def run_catalog_scan(job_id: str, prefix: str, remove_missing: bool,
                     settings: Settings) -> None:
    job = scan_jobs[job_id]
    try:
        job.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
        runtime.store.save_scan_job(job)
        extensions = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}
        roots = configured_catalog_roots(settings)
        scan_targets: list[tuple[Path, Path]] = []
        for root in roots:
            target = (root / prefix).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"扫描目录越界：{root}")
            if target.is_dir():
                scan_targets.append((root, target))
        if not scan_targets:
            raise ValueError(f"目录不存在：{prefix or '/'}")

        files: list[tuple[Path, Path]] = []
        for root, target in scan_targets:
            files.extend((root, path) for path in target.rglob("*")
                         if path.is_file() and path.suffix.lower() in extensions
                         and not path.stem.lower().endswith(("_thumb", "_thumbnail")))
        files.sort(key=lambda item: item[1].as_posix())
        normalized_prefix = "" if not prefix else prefix.strip("/\\").replace("\\", "/").rstrip("/") + "/"
        existing = runtime.store.current_signatures(settings.model_version, normalized_prefix)
        seen_file_ids: set[str] = set()
        pending: list[tuple[Path, Path, str, str, str, str, int, int]] = []
        for root, path in files:
            relative = path.relative_to(root).as_posix()
            file_id = catalog_file_id(path, root)
            category = catalog_category(relative)
            seen_file_ids.add(file_id)
            try:
                stat = path.stat()
                signature = f"{stat.st_size}:{stat.st_mtime_ns}"
                if existing.get(file_id) == signature:
                    job["skipped"] = int(job["skipped"]) + 1
                    continue
                pending.append((root, path, relative, file_id, category, signature,
                                stat.st_size, stat.st_mtime_ns))
            except OSError as exc:
                job["failed"] = int(job["failed"]) + 1
                logger.warning("Cannot inspect catalog image %s: %s", relative, exc)
        job.update(total=len(files))
        runtime.store.save_scan_job(job)

        for start in range(0, len(pending), settings.max_batch_size):
            batch_paths = pending[start:start + settings.max_batch_size]
            decoded = []
            metadata: list[tuple[Path, Path, str, str, str, str, int, int]] = []
            for root, path, relative, file_id, category, signature, file_size, modified_ns in batch_paths:
                job["current"] = relative
                try:
                    decoded.append(decode_image(path.read_bytes(), settings.max_image_pixels))
                    metadata.append((root, path, relative, file_id, category, signature,
                                     file_size, modified_ns))
                except (InvalidImageError, OSError) as exc:
                    job["failed"] = int(job["failed"]) + 1
                    logger.warning("Skipping catalog image %s: %s", relative, exc)
            if not decoded:
                continue
            try:
                vectors = runtime.embedder.embed(decoded)
                items = [IndexItem(
                    product_id=f"catalog:{file_id}",
                    product_code=path.name,
                    file_id=file_id,
                    category=category,
                    image_url=f"{settings.public_image_base_url}{quote(relative, safe='')}",
                    vector=vector,
                    signature=signature,
                    file_size=file_size,
                    modified_ns=modified_ns,
                ) for (_root, path, relative, file_id, category, signature, file_size, modified_ns), vector
                    in zip(metadata, vectors)]
                runtime.store.upsert_images(items, settings.model_version)
                job["indexed"] = int(job["indexed"]) + len(items)
                runtime.store.save_scan_job(job)
            except Exception:
                job["failed"] = int(job["failed"]) + len(decoded)
                logger.exception("Catalog batch failed at %s", job.get("current"))
                runtime.store.save_scan_job(job)
        if remove_missing:
            missing = sorted(set(existing) - seen_file_ids)
            job["removed"] = runtime.store.delete_file_ids(missing, settings.model_version)
        job.update(
            status="completed",
            current=None,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        runtime.store.save_scan_job(job)
    except Exception as exc:
        logger.exception("Catalog scan failed")
        job.update(
            status="failed",
            error=str(exc),
            current=None,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        runtime.store.save_scan_job(job)


def catalog_file_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parent = path.parent.name
    normalized = parent.replace("-", "").replace("_", "")
    if len(normalized) >= 12 and normalized.isalnum():
        return parent
    return relative


def catalog_category(relative: str) -> str:
    normalized = "/" + relative.lower().replace("\\", "/")
    if "/devsourcing/" in normalized or "/product-dev" in normalized:
        return "development"
    if normalized.startswith("/product/"):
        return "product"
    return relative.split("/", 1)[0].lower()


@app.post("/v1/catalog/scan", response_model=CatalogScanResponse,
          dependencies=[Depends(require_api_key)])
async def start_catalog_scan(request: CatalogScanRequest,
                             settings: Settings = Depends(get_settings)) -> CatalogScanResponse:
    running = next((item for item in scan_jobs.values()
                    if item["status"] in {"queued", "running"}), None)
    if running is not None:
        return CatalogScanResponse(**running)
    job_id = str(uuid4())
    job: dict[str, object] = {
        "job_id": job_id, "status": "queued", "total": 0,
        "indexed": 0, "skipped": 0, "failed": 0, "removed": 0,
        "current": None, "error": None, "started_at": None, "completed_at": None,
        "prefix": request.prefix, "remove_missing": request.remove_missing,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    scan_jobs[job_id] = job
    await run_in_threadpool(runtime.store.save_scan_job, job)
    task = asyncio.create_task(run_in_threadpool(
        run_catalog_scan, job_id, request.prefix, request.remove_missing, settings))
    scan_tasks.add(task)
    task.add_done_callback(scan_tasks.discard)
    return CatalogScanResponse(**job)


@app.get("/v1/catalog/stats", response_model=CatalogStatsResponse,
         dependencies=[Depends(require_api_key)])
async def catalog_stats(settings: Settings = Depends(get_settings)) -> CatalogStatsResponse:
    result = await run_in_threadpool(runtime.store.stats, settings.model_version)
    active = next((CatalogScanResponse(**item) for item in scan_jobs.values()
                   if item["status"] in {"queued", "running"}), None)
    latest_data = await run_in_threadpool(runtime.store.latest_scan_job)
    latest = CatalogScanResponse(**latest_data) if latest_data else None
    return CatalogStatsResponse(**result, active_job=active, latest_job=latest)


@app.get("/v1/catalog/assets", response_model=CatalogAssetsResponse,
         dependencies=[Depends(require_api_key)])
async def catalog_assets(
    offset: int = 0,
    limit: int = 50,
    settings: Settings = Depends(get_settings),
) -> CatalogAssetsResponse:
    if offset < 0 or limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="offset or limit is out of range")
    items = await run_in_threadpool(
        runtime.store.list_assets, settings.model_version, offset, limit)
    stats = await run_in_threadpool(runtime.store.stats, settings.model_version)
    return CatalogAssetsResponse(
        offset=offset, limit=limit, total=int(stats["indexed_count"]), items=items)


@app.get("/v1/catalog/scan/latest", response_model=CatalogScanResponse | None,
         dependencies=[Depends(require_api_key)])
async def latest_catalog_scan() -> CatalogScanResponse | None:
    job = await run_in_threadpool(runtime.store.latest_scan_job)
    return CatalogScanResponse(**job) if job else None


@app.get("/v1/catalog/scan/{job_id}", response_model=CatalogScanResponse,
         dependencies=[Depends(require_api_key)])
async def catalog_scan_status(job_id: str) -> CatalogScanResponse:
    job = scan_jobs.get(job_id)
    if job is None:
        job = await run_in_threadpool(runtime.store.get_scan_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="扫描任务不存在")
    return CatalogScanResponse(**job)


@app.get("/v1/catalog/image", include_in_schema=False)
async def catalog_image(path: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    for root in configured_catalog_roots(settings):
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            continue
        if target.is_file():
            return FileResponse(target)
    raise HTTPException(status_code=404, detail="图片不存在")


@app.delete("/v1/index/products/{product_id}", status_code=204,
            dependencies=[Depends(require_api_key)])
async def delete_product(product_id: str) -> None:
    await run_in_threadpool(runtime.store.delete_product, product_id)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    vector_ready = False
    if runtime.store is not None:
        try:
            vector_ready = await run_in_threadpool(runtime.store.ready)
        except Exception:
            logger.exception("Qdrant readiness check failed")
    loaded = runtime.embedder is not None
    if not loaded or not vector_ready:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return {
        "status": "ok",
        "model_loaded": loaded,
        "vector_store_ready": vector_ready,
        "model_version": settings.model_version,
    }
