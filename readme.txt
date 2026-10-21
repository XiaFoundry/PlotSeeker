PlotSeeker/
├── data/
│   ├── raw/                 # 原始 JSONL，不修改
│   └── processed/           # 清洗后的中间数据
├── storage/
│   ├── plotseeker.db        # SQLite
│   └── vector_store/        # 持久化向量索引
├── scripts/
│   ├── audit_data.py
│   ├── import_sqlite.py
│   ├── build_documents.py
│   └── build_index.py
├── src/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── evaluation/
│   ├── agent/
│   └── api/
├── tests/
├── benchmarks/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
└── README.md

注意！所有的临时脚本，也应该放在一个废弃文件夹内，以记录工作流内容。