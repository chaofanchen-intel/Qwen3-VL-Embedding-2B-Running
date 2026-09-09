#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaofan/model_benchmark}
QWEN3=${QWEN3:-$ROOT/Qwen3-VL-Embedding-2B-Running}
INSTALL=${INSTALL:-$ROOT/benchmark/openvino/install}
PYTHON=${PYTHON:-$HOME/envs/qwen_env/bin/python3}
PRECISIONS=${PRECISIONS:-INT8,INT4}

cd "$QWEN3"
exec env \
  PYTHONPATH="$INSTALL/python:$QWEN3" \
  LD_LIBRARY_PATH="$INSTALL/runtime/3rdparty/tbb/lib:$INSTALL/runtime/lib/intel64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OPENVINO_LIB_PATHS="$INSTALL/runtime/lib/intel64" \
  OPENVINO_PLUGIN_PATH="$INSTALL/runtime/lib/intel64" \
  "$PYTHON" -u mmdocir_openvino_eval_qwen3/run_balanced_200.py \
    --precisions "$PRECISIONS" "$@"
