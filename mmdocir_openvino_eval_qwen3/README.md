# Qwen3-VL-Embedding-2B MMDocIR balanced_200 evaluation

This is a self-contained MMDocIR evaluator for `Qwen3-VL-Embedding-2B`. It owns its
runner, subset manifest, cache, per-query files, and result summaries inside this
repository. It does not import code, cache, or result files from another model
repository.

## Scope

- Dataset: MMDocIR Evaluation Dataset, revision `bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4`
- Subset: local `subsets/balanced_200.json`, 200 queries, 34 documents, 607 pages, 7,956 layouts
- Device: GPU
- Precision: Qwen3-VL-Embedding-2B-INT8 and INT4
- Native output: 2048 dimensions
- Query/document prompt: `Represent the user's input.`
- Page images: source dataset `image_binary`
- Layouts: text/equation use text embedding; image/table use paired fused embedding
- Search scopes: close-domain and open-domain

The Qwen3 multimodal API cannot express paired `(text_i, image_i)` batches. Page and
image/table layout items therefore run one paired `EmbeddingPipeline.embed()` call per
item. Text queries and text layouts use true text batches.

## Runtime

The Qwen3 GenAI fix must be built and installed before running the evaluator. The current
OpenVINO GPU plugin may contain the separate GPU optimizations used by another workload;
this evaluator does not import that workload's code. The Qwen3 GenAI patch is the required
fix for the Qwen3 language extra-input shapes and multimodal batch handling.

```bash
ROOT=/home/chaofan/model_benchmark
GENAI=$ROOT/benchmark/openvino.genai
INSTALL=$ROOT/benchmark/openvino/install

cd "$GENAI"
cmake --build build -j2
cmake --install build --config Release --prefix "$INSTALL"
```

Use the install tree consistently:

```bash
cd /home/chaofan/model_benchmark/Qwen3-VL-Embedding-2B-Running
PYTHONPATH=/home/chaofan/model_benchmark/benchmark/openvino/install/python \
LD_LIBRARY_PATH=/home/chaofan/model_benchmark/benchmark/openvino/install/runtime/3rdparty/tbb/lib:/home/chaofan/model_benchmark/benchmark/openvino/install/runtime/lib/intel64 \
OPENVINO_LIB_PATHS=/home/chaofan/model_benchmark/benchmark/openvino/install/runtime/lib/intel64 \
OPENVINO_PLUGIN_PATH=/home/chaofan/model_benchmark/benchmark/openvino/install/runtime/lib/intel64 \
/home/chaofan/envs/qwen_env/bin/python3 -u mmdocir_openvino_eval_qwen3/run_balanced_200.py \
  --precisions INT8,INT4
```

The default dataset path is `/home/chaofan/datasets/MMDocIR_Evaluation_Dataset`. Override
`--dataset-dir` if needed. Results and caches are written only below
`mmdocir_openvino_eval_qwen3/results/balanced_200/`.

## Outputs

- `summary.json`: runtime, subset, cache timing, close/open page and layout metrics
- `int8/summary.json`, `int4/summary.json`: per-precision summaries
- `int8/cache/`, `int4/cache/`: normalized 2048-dimensional vectors and timing metadata
- `int8/per_query_close.jsonl`, `int8/per_query_open.jsonl`: per-query metrics
- `int4/per_query_close.jsonl`, `int4/per_query_open.jsonl`: per-query metrics

Readable report: [`REPORT.md`](REPORT.md)

A zero-area ground-truth layout is excluded from layout aggregation and recorded in
`layout_exclusions`; page metrics remain unaffected. This follows the dataset's metric
semantics without inventing an area for an invalid denominator.
