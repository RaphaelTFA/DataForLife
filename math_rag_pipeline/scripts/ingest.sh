#!/bin/bash
set -e

echo "🚀 [Ingest] Bắt đầu ingest dữ liệu PDF trong data/raw/..."
for pdf in data/raw/*.pdf; do
  echo "📘 Đang xử lý $pdf ..."
  python -m rag_toan.cli ingest "$pdf"
done

echo "✅ Hoàn tất ingest tất cả PDF → ./data/extracted/"
