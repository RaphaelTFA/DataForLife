#!/bin/bash
set -e

echo "🔍 [Index] Đang build vector index..."
python -m rag_toan.cli index
echo "✅ Index đã lưu trong ./data/vector_db"
