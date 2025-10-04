#!/usr/bin/env bash
# Usage: ./hw2_seq2seq.sh <data_dir> <output_txt>

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "USAGE: $0 <data_dir> <output_txt>"
  exit 1
fi

DATA_DIR="../../MLDS_hw2_1_data"
OUT_TXT="testset_output.txt"

CKPT="${CKPT:-seq2seq_model/ep199.pt}"
IDS_TXT="$DATA_DIR/testing_data/id.txt"
TEST_LABEL_JSON="$DATA_DIR/testing_label.json"

echo "==> Running inference..."
python model_seq2seq.py infer \
  --data_dir "$DATA_DIR" \
  --ids_txt "$IDS_TXT" \
  --ckpt_path "$CKPT" \
  --out_txt "$OUT_TXT" \
  --beam 5 \
  --max_len 30 \
  --device cuda
