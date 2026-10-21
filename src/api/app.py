"""FastAPI wrapper around the P5 retrieval service.

The model and indexes are process-level singletons; the first semantic query
may pay the model-loading cost, later queries reuse the same objects.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import MODEL_CACHE_DIR, SQLITE_PATH, VECTOR_STORE_DIR
from retrieval.keyword_index import KEYWORD_DB_PATH, KeywordIndexError
from retrieval.search import MAX_QUERY_CHARS, RetrievalService
from retrieval.vector_search import VectorIndexError

API_VERSION = "p8-v1"
_SEARCH_TIMEOUT_DEFAULT = 30.0
_service: RetrievalService | None = None
_service_lock = threading.Lock()


class HealthResponse(BaseModel):
    status: str
    api_version: str
    keyword_index: bool
    vector_index: bool
    model_cache: bool
    sqlite: bool


class SearchResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    candidate_k: int
    rrf_k: int
    confidence: str
    warnings: list[str] = Field(default_factory=list)
    keyword_coverage: float
    vector_top_similarity: float | None = None
    counts: dict[str, int]
    timings_ms: dict[str, float]
    candidates: list[dict[str, Any]]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _service_paths() -> dict[str, Path]:
    return {
        "keyword_db": _env_path("PLOTSEEKER_KEYWORD_DB", KEYWORD_DB_PATH),
        "store_dir": _env_path("PLOTSEEKER_VECTOR_STORE", VECTOR_STORE_DIR),
        "model_cache": _env_path("PLOTSEEKER_MODEL_CACHE", MODEL_CACHE_DIR / "bge-m3"),
        "sqlite": _env_path("PLOTSEEKER_SQLITE", SQLITE_PATH),
    }


def get_service() -> RetrievalService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                paths = _service_paths()
                _service = RetrievalService(
                    keyword_db=paths["keyword_db"],
                    store_dir=paths["store_dir"],
                    model_cache=paths["model_cache"],
                    sqlite_path=paths["sqlite"],
                    device=os.environ.get("PLOTSEEKER_DEVICE", "auto"),
                )
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_timeout_seconds = float(
        os.environ.get("PLOTSEEKER_SEARCH_TIMEOUT_SECONDS", _SEARCH_TIMEOUT_DEFAULT)
    )
    yield
    global _service
    if _service is not None:
        _service.close()
        _service = None


app = FastAPI(
    title="PlotSeeker API",
    version=API_VERSION,
    description="Fuzzy plot / character / episode retrieval over Bangumi Archive.",
    lifespan=lifespan,
)
app.state.search_timeout_seconds = _SEARCH_TIMEOUT_DEFAULT


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    paths = _service_paths()
    model_cache = paths["model_cache"]
    checkpoint_path = paths["store_dir"] / "p4_checkpoint.json"
    vector_ready = False
    if checkpoint_path.is_file():
        try:
            vector_ready = (
                json.loads(checkpoint_path.read_text(encoding="utf-8")).get("state")
                == "complete"
            )
        except (OSError, json.JSONDecodeError):
            vector_ready = False
    checks = {
        "keyword_index": paths["keyword_db"].is_file(),
        "vector_index": vector_ready,
        "model_cache": (model_cache / "config.json").is_file(),
        "sqlite": paths["sqlite"].is_file(),
    }
    return HealthResponse(
        status="ok" if all(checks.values()) else "degraded",
        api_version=API_VERSION,
        **checks,
    )


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS),
    top_k: int = Query(10, ge=1, le=50),
    mode: Literal["keyword", "vector", "hybrid"] = "hybrid",
    candidate_k: int = Query(50, ge=1, le=200),
    service: RetrievalService = Depends(get_service),
) -> SearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is empty")
    timeout = float(getattr(app.state, "search_timeout_seconds", _SEARCH_TIMEOUT_DEFAULT))
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                service.search,
                query,
                mode=mode,
                top_k=top_k,
                candidate_k=candidate_k,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail=f"search timed out after {timeout}s"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, KeywordIndexError, VectorIndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - API boundary
        raise HTTPException(status_code=500, detail="internal retrieval error") from exc
    return SearchResponse(**result)
