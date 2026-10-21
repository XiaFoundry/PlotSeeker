"""Project-wide paths and defaults.

Every formal script imports paths from here so that no machine-specific
absolute path is hardcoded outside this module.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENTS_DIR = PROCESSED_DIR / "documents"

STORAGE_DIR = PROJECT_ROOT / "storage"
VECTOR_STORE_DIR = STORAGE_DIR / "vector_store"
MODEL_CACHE_DIR = STORAGE_DIR / "model_cache"
SQLITE_PATH = STORAGE_DIR / "plotseeker.db"
IMPORT_REPORT_PATH = STORAGE_DIR / "import-report.json"
INDEX_CHECKPOINT_PATH = VECTOR_STORE_DIR / "p4_checkpoint.json"
INDEX_REPORT_PATH = VECTOR_STORE_DIR / "p4_report.json"

CONFIG_DIR = PROJECT_ROOT / "config"
AGENT_CONFIG_PATH = CONFIG_DIR / "agent.json"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

MAIN_ENTITY_FILES = {
    "subject": "subject.jsonlines",
    "character": "character.jsonlines",
    "person": "person.jsonlines",
    "episode": "episode.jsonlines",
}

RELATION_FILES = {
    "subject-characters": "subject-characters.jsonlines",
    "subject-persons": "subject-persons.jsonlines",
    "subject-relations": "subject-relations.jsonlines",
    "person-characters": "person-characters.jsonlines",
    "person-relations": "person-relations.jsonlines",
}

ALL_DATA_FILES = list(MAIN_ENTITY_FILES.values()) + list(RELATION_FILES.values())
