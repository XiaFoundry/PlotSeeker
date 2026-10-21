#!/usr/bin/env python
"""Start the PlotSeeker FastAPI service with the existing indexes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MODEL_CACHE_DIR, SQLITE_PATH, VECTOR_STORE_DIR  # noqa: E402
from retrieval.keyword_index import KEYWORD_DB_PATH  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PlotSeeker FastAPI service.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--keyword-db", default=str(KEYWORD_DB_PATH), help="P5 FTS5 database."
    )
    parser.add_argument(
        "--store-dir", default=str(VECTOR_STORE_DIR), help="P4 Chroma directory."
    )
    parser.add_argument(
        "--model-cache",
        default=str(MODEL_CACHE_DIR / "bge-m3"),
        help="BGE-M3 snapshot directory.",
    )
    parser.add_argument(
        "--sqlite", default=str(SQLITE_PATH), help="P2 SQLite database."
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request search timeout in seconds.",
    )
    parser.add_argument(
        "--log-level", default="info", help="Uvicorn log level."
    )
    args = parser.parse_args(argv)

    os.environ["PLOTSEEKER_KEYWORD_DB"] = str(Path(args.keyword_db))
    os.environ["PLOTSEEKER_VECTOR_STORE"] = str(Path(args.store_dir))
    os.environ["PLOTSEEKER_MODEL_CACHE"] = str(Path(args.model_cache))
    os.environ["PLOTSEEKER_SQLITE"] = str(Path(args.sqlite))
    os.environ["PLOTSEEKER_DEVICE"] = args.device
    os.environ["PLOTSEEKER_SEARCH_TIMEOUT_SECONDS"] = str(args.timeout)

    import uvicorn

    from api.app import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
