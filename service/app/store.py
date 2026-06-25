from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

import numpy as np

try:
    from qdrant_client import QdrantClient, models
except ImportError:  # SQLite backend does not require qdrant-client.
    QdrantClient = None
    models = None

from app.config import Settings


@dataclass(frozen=True)
class Evidence:
    file_id: str
    image_url: str | None
    score: float


@dataclass(frozen=True)
class ProductResult:
    product_id: str
    product_code: str
    score: float
    evidence: list[Evidence]


@dataclass(frozen=True)
class IndexItem:
    product_id: str
    product_code: str
    file_id: str
    category: str | None
    image_url: str | None
    vector: list[float]
    signature: str = ""
    file_size: int = 0
    modified_ns: int = 0


def _point_id(model_version: str, file_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{model_version}:{file_id}"))


class VectorStore:
    """Optional Qdrant backend for larger or distributed deployments."""

    def __init__(self, settings: Settings, vector_size: int):
        if QdrantClient is None or models is None:
            raise RuntimeError("Qdrant backend requires qdrant-client")
        self._settings = settings
        self._client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30
        )
        self._collection = settings.collection_name
        self._scan_jobs: dict[str, dict] = {}
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE, on_disk=True
            ),
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200, on_disk=True),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=False
                )
            ),
        )
        for field in ("product_id", "category", "model_version", "file_id"):
            self._client.create_payload_index(
                self._collection,
                field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def upsert_image(
        self,
        *,
        product_id: str,
        product_code: str,
        file_id: str,
        category: str | None,
        image_url: str | None,
        vector: list[float],
        model_version: str,
        signature: str = "",
        file_size: int = 0,
        modified_ns: int = 0,
    ) -> str:
        item = IndexItem(
            product_id, product_code, file_id, category, image_url, vector,
            signature, file_size, modified_ns,
        )
        return self.upsert_images([item], model_version)[0]

    def upsert_images(self, items: list[IndexItem], model_version: str) -> list[str]:
        now = datetime.now(timezone.utc).isoformat()
        point_ids = [_point_id(model_version, item.file_id) for item in items]
        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=item.vector,
                    payload={
                        "product_id": item.product_id,
                        "product_code": item.product_code,
                        "file_id": item.file_id,
                        "category": item.category or "",
                        "image_url": item.image_url,
                        "model_version": model_version,
                        "signature": item.signature,
                        "file_size": item.file_size,
                        "modified_ns": item.modified_ns,
                        "updated_at": now,
                    },
                )
                for point_id, item in zip(point_ids, items)
            ],
            wait=True,
        )
        return point_ids

    def current_signatures(self, model_version: str, prefix: str = "") -> dict[str, str]:
        result: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="model_version", match=models.MatchValue(value=model_version)
                )]),
                limit=1000,
                offset=offset,
                with_payload=["file_id", "signature"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                file_id = str(payload.get("file_id", ""))
                if file_id and (not prefix or file_id.startswith(prefix)):
                    result[file_id] = str(payload.get("signature", ""))
            if offset is None:
                break
        return result

    def get_signature(self, file_id: str, model_version: str) -> str | None:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[_point_id(model_version, file_id)],
            with_payload=["signature"],
            with_vectors=False,
        )
        if not points:
            return None
        return str((points[0].payload or {}).get("signature", ""))

    def search(self, vectors: list[list[float]], top_k: int,
               category: str | None) -> list[ProductResult]:
        limit = min(max(top_k * self._settings.candidate_multiplier, 50), 1000)
        query_filter = None
        if category:
            query_filter = models.Filter(must=[models.FieldCondition(
                key="category", match=models.MatchValue(value=category)
            )])
        best_points: dict[str, tuple[float, dict]] = {}
        for vector in vectors:
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                search_params=models.SearchParams(hnsw_ef=256, exact=False),
            )
            for point in response.points:
                key = str(point.id)
                value = (float(point.score), point.payload or {})
                if key not in best_points or value[0] > best_points[key][0]:
                    best_points[key] = value
        return _aggregate_products(list(best_points.values()), top_k)

    def delete_product(self, product_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(
                    key="product_id", match=models.MatchValue(value=product_id)
                )
            ])),
            wait=True,
        )

    def delete_file_ids(self, file_ids: list[str], model_version: str) -> int:
        if not file_ids:
            return 0
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(
                points=[_point_id(model_version, file_id) for file_id in file_ids]
            ),
            wait=True,
        )
        return len(file_ids)

    def stats(self, model_version: str) -> dict:
        info = self._client.get_collection(self._collection)
        return {
            "backend": "qdrant",
            "location": self._settings.qdrant_url,
            "collection": self._collection,
            "model_version": model_version,
            "indexed_count": int(info.points_count or 0),
            "database_bytes": None,
        }

    def list_assets(self, model_version: str, offset: int, limit: int) -> list[dict]:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="model_version", match=models.MatchValue(value=model_version)
            )]),
            limit=min(offset + limit, 1000),
            with_payload=True,
            with_vectors=False,
        )
        return [point.payload or {} for point in points[offset:offset + limit]]

    def save_scan_job(self, job: dict) -> None:
        self._scan_jobs[str(job["job_id"])] = dict(job)

    def get_scan_job(self, job_id: str) -> dict | None:
        job = self._scan_jobs.get(job_id)
        return None if job is None else dict(job)

    def latest_scan_job(self) -> dict | None:
        if not self._scan_jobs:
            return None
        return dict(list(self._scan_jobs.values())[-1])

    def ready(self) -> bool:
        self._client.get_collection(self._collection)
        return True


class SqliteVectorStore:
    """Durable incremental exact-cosine store for a single CPU server."""

    def __init__(self, settings: Settings, vector_size: int):
        self._settings = settings
        self._vector_size = vector_size
        configured = settings.sqlite_path or settings.local_store_path
        path = Path(configured)
        if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            path = path.with_suffix(".db")
        self._path = path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: list[dict] = []
        self._matrix = np.empty((0, vector_size), dtype=np.float32)
        self._cache_dirty = False
        self._initialize()
        self._reload_cache()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS visual_assets (
                    point_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    product_code TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    image_url TEXT,
                    model_version TEXT NOT NULL,
                    signature TEXT NOT NULL DEFAULT '',
                    file_size INTEGER NOT NULL DEFAULT 0,
                    modified_ns INTEGER NOT NULL DEFAULT 0,
                    vector BLOB NOT NULL,
                    vector_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(model_version, file_id)
                );
                CREATE INDEX IF NOT EXISTS ix_visual_assets_product
                    ON visual_assets(model_version, product_id);
                CREATE INDEX IF NOT EXISTS ix_visual_assets_category
                    ON visual_assets(model_version, category);
                CREATE INDEX IF NOT EXISTS ix_visual_assets_updated
                    ON visual_assets(updated_at DESC);
                CREATE TABLE IF NOT EXISTS visual_scan_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    prefix TEXT NOT NULL DEFAULT '',
                    remove_missing INTEGER NOT NULL DEFAULT 1,
                    total INTEGER NOT NULL DEFAULT 0,
                    indexed INTEGER NOT NULL DEFAULT 0,
                    skipped INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    removed INTEGER NOT NULL DEFAULT 0,
                    current TEXT,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_visual_scan_jobs_updated
                    ON visual_scan_jobs(updated_at DESC);
                """
            )
            connection.execute(
                """UPDATE visual_scan_jobs
                   SET status='failed', error='服务重启导致任务中断，请重新执行同步',
                       completed_at=?, updated_at=?
                   WHERE status IN ('queued', 'running')""",
                (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
            )

    def _reload_cache(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT product_id, product_code, file_id, category, image_url,
                          model_version, signature, file_size, modified_ns, vector,
                          updated_at
                   FROM visual_assets
                   WHERE model_version = ? ORDER BY rowid""",
                (self._settings.model_version,),
            ).fetchall()
        records: list[dict] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.shape != (self._vector_size,):
                raise RuntimeError(
                    "SQLite vector dimension is incompatible with the active model version"
                )
            records.append({key: row[key] for key in row.keys() if key != "vector"})
            vectors.append(vector)
        with self._lock:
            self._records = records
            self._matrix = (
                np.stack(vectors).astype(np.float32, copy=False)
                if vectors else np.empty((0, self._vector_size), dtype=np.float32)
            )
            self._cache_dirty = False

    def upsert_image(
        self,
        *,
        product_id: str,
        product_code: str,
        file_id: str,
        category: str | None,
        image_url: str | None,
        vector: list[float],
        model_version: str,
        signature: str = "",
        file_size: int = 0,
        modified_ns: int = 0,
    ) -> str:
        item = IndexItem(
            product_id, product_code, file_id, category, image_url, vector,
            signature, file_size, modified_ns,
        )
        return self.upsert_images([item], model_version)[0]

    def upsert_images(self, items: list[IndexItem], model_version: str) -> list[str]:
        if not items:
            return []
        now = datetime.now(timezone.utc).isoformat()
        values = []
        point_ids = []
        for item in items:
            vector = np.asarray(item.vector, dtype=np.float32)
            if vector.shape != (self._vector_size,):
                raise ValueError("Incorrect vector dimension")
            point_id = _point_id(model_version, item.file_id)
            point_ids.append(point_id)
            values.append((
                point_id, item.product_id, item.product_code, item.file_id,
                item.category or "", item.image_url, model_version, item.signature,
                item.file_size, item.modified_ns, vector.tobytes(), self._vector_size,
                now, now,
            ))
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO visual_assets (
                       point_id, product_id, product_code, file_id, category, image_url,
                       model_version, signature, file_size, modified_ns, vector,
                       vector_size, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(model_version, file_id) DO UPDATE SET
                       point_id=excluded.point_id,
                       product_id=excluded.product_id,
                       product_code=excluded.product_code,
                       category=excluded.category,
                       image_url=excluded.image_url,
                       signature=excluded.signature,
                       file_size=excluded.file_size,
                       modified_ns=excluded.modified_ns,
                       vector=excluded.vector,
                       vector_size=excluded.vector_size,
                       updated_at=excluded.updated_at""",
                values,
            )
        with self._lock:
            self._cache_dirty = True
        return point_ids

    def current_signatures(self, model_version: str, prefix: str = "") -> dict[str, str]:
        sql = "SELECT file_id, signature FROM visual_assets WHERE model_version = ?"
        parameters: list[object] = [model_version]
        if prefix:
            sql += " AND file_id LIKE ? ESCAPE '\\'"
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            parameters.append(f"{escaped}%")
        with self._connect() as connection:
            return {
                str(row["file_id"]): str(row["signature"])
                for row in connection.execute(sql, parameters)
            }

    def get_signature(self, file_id: str, model_version: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT signature FROM visual_assets WHERE model_version = ? AND file_id = ?",
                (model_version, file_id),
            ).fetchone()
        return None if row is None else str(row["signature"])

    def search(self, vectors: list[list[float]], top_k: int,
               category: str | None) -> list[ProductResult]:
        if self._cache_dirty:
            self._reload_cache()
        with self._lock:
            if not self._records:
                return []
            if category:
                indices = [i for i, record in enumerate(self._records)
                           if record.get("category") == category]
                if not indices:
                    return []
                matrix = self._matrix[indices]
                records = [self._records[i] for i in indices]
            else:
                matrix = self._matrix
                records = self._records
            queries = np.asarray(vectors, dtype=np.float32)
            scores = np.max(queries @ matrix.T, axis=0)
            limit = min(len(records), max(top_k * self._settings.candidate_multiplier, 50))
            candidate_indices = np.argpartition(scores, len(scores) - limit)[-limit:]
            points = sorted(
                ((float(scores[i]), records[i]) for i in candidate_indices),
                reverse=True,
                key=lambda item: item[0],
            )
        return _aggregate_products(points, top_k)

    def delete_product(self, product_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM visual_assets WHERE model_version = ? AND product_id = ?",
                (self._settings.model_version, product_id),
            )
        with self._lock:
            self._cache_dirty = True

    def delete_file_ids(self, file_ids: list[str], model_version: str) -> int:
        if not file_ids:
            return 0
        deleted = 0
        with self._connect() as connection:
            for start in range(0, len(file_ids), 500):
                chunk = file_ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor = connection.execute(
                    f"DELETE FROM visual_assets WHERE model_version = ? "
                    f"AND file_id IN ({placeholders})",
                    [model_version, *chunk],
                )
                deleted += cursor.rowcount
        if deleted:
            with self._lock:
                self._cache_dirty = True
        return deleted

    def stats(self, model_version: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS count, COALESCE(SUM(file_size), 0) AS source_bytes,
                          MAX(updated_at) AS last_updated_at
                   FROM visual_assets WHERE model_version = ?""",
                (model_version,),
            ).fetchone()
        database_bytes = sum(
            candidate.stat().st_size for candidate in (
                self._path,
                Path(f"{self._path}-wal"),
                Path(f"{self._path}-shm"),
            ) if candidate.exists()
        )
        return {
            "backend": "sqlite",
            "location": str(self._path),
            "collection": self._settings.collection_name,
            "model_version": model_version,
            "indexed_count": int(row["count"]),
            "source_bytes": int(row["source_bytes"]),
            "database_bytes": database_bytes,
            "last_updated_at": row["last_updated_at"],
            "vector_dimension": self._vector_size,
        }

    def list_assets(self, model_version: str, offset: int, limit: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT product_id, product_code, file_id, category, image_url,
                          signature, file_size, modified_ns, updated_at
                   FROM visual_assets WHERE model_version = ?
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (model_version, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_scan_job(self, job: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO visual_scan_jobs (
                       job_id, status, prefix, remove_missing, total, indexed, skipped,
                       failed, removed, current, error, started_at, completed_at,
                       created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                       status=excluded.status, prefix=excluded.prefix,
                       remove_missing=excluded.remove_missing, total=excluded.total,
                       indexed=excluded.indexed, skipped=excluded.skipped,
                       failed=excluded.failed, removed=excluded.removed,
                       current=excluded.current, error=excluded.error,
                       started_at=excluded.started_at, completed_at=excluded.completed_at,
                       updated_at=excluded.updated_at""",
                (
                    job["job_id"], job["status"], job.get("prefix", ""),
                    1 if job.get("remove_missing", False) else 0,
                    job.get("total", 0), job.get("indexed", 0), job.get("skipped", 0),
                    job.get("failed", 0), job.get("removed", 0), job.get("current"),
                    job.get("error"), job.get("started_at"), job.get("completed_at"),
                    job.get("created_at", now), now,
                ),
            )

    def get_scan_job(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM visual_scan_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._scan_job_dict(row)

    def latest_scan_job(self) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM visual_scan_jobs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self._scan_job_dict(row)

    @staticmethod
    def _scan_job_dict(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        result["remove_missing"] = bool(result["remove_missing"])
        result.pop("updated_at", None)
        return result

    def ready(self) -> bool:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return True


def _aggregate_products(points: list[tuple[float, dict]], top_k: int) -> list[ProductResult]:
    grouped: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for score, payload in points:
        grouped[str(payload.get("product_id", ""))].append((score, payload))
    results = []
    for product_id, product_points in grouped.items():
        if not product_id:
            continue
        product_points.sort(key=lambda item: item[0], reverse=True)
        top = product_points[:3]
        score = 0.9 * top[0][0] + 0.1 * (
            sum(item[0] for item in top) / len(top)
        )
        results.append(ProductResult(
            product_id=product_id,
            product_code=str(top[0][1].get("product_code", "")),
            score=round(score, 6),
            evidence=[Evidence(
                str(payload.get("file_id", "")),
                payload.get("image_url"),
                round(point_score, 6),
            ) for point_score, payload in top],
        ))
    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def create_vector_store(settings: Settings, vector_size: int):
    backend = settings.store_backend.lower()
    if backend == "qdrant":
        return VectorStore(settings, vector_size)
    if backend in {"local", "sqlite"}:
        return SqliteVectorStore(settings, vector_size)
    raise ValueError(f"Unsupported vector store backend: {settings.store_backend}")
