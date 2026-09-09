# Qwen3-VL-Embedding-2B MMDocIR balanced_200 report

日期：2026-09-07

## 运行范围

本报告来自 Qwen3 repo 内独立的
[`mmdocir_openvino_eval_qwen3/`](mmdocir_openvino_eval_qwen3/) evaluator。它不导入
其它模型 repo 的代码、cache 或结果。

| 项目 | 值 |
| --- | --- |
| Dataset | MMDocIR Evaluation Dataset |
| Dataset revision | `bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4` |
| Subset | local `subsets/balanced_200.json` |
| Queries | 200 |
| Domains | 10 |
| Documents | 34 |
| Pages | 607 |
| Layouts | 7,956 |
| Device | GPU |
| Output | native 2048 dimensions |
| OpenVINO | `2026.3.0-22455-2080527e54-qwen-ttft-optimizations` |
| GenAI | patched Qwen3 build `2026.3.0.0-3278-0c38b54d7e7-qwen-ttft-llm-bench` |
| oneDNN primitive cache | `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` |

The Qwen3 GenAI patch was required for the language extra-input shape fix and multimodal
batch handling. The existing OpenVINO GPU patch tree remained in place; text/image/fused
smoke returned `[1,2048]` for all three modes after the patched GenAI install.

Image/table layout items use one paired Qwen3 multimodal call per item. Text queries and
text/equation layouts use true text batches. No ground-truth layouts were excluded in this
balanced subset.

Raw result: [`results/balanced_200/summary.json`](results/balanced_200/summary.json).
Per-precision summaries and normalized vector caches are under `results/balanced_200/int8/`
and `results/balanced_200/int4/`.

## Quality metrics

Values are macro averages over 200 queries. `Recall@K` and `nDCG@K` are reported as
`@5 / @10`. Layout metrics use graded bbox intersection for nDCG and area accumulation for
Recall, matching the independent Qwen3 evaluator's documented definition.

| precision | scope | page Recall@5/10 | page nDCG@5/10 | layout Recall@5/10 | layout nDCG@5/10 |
| --- | --- | ---: | ---: | ---: | ---: |
| INT8 | close | 0.8308 / 0.8933 | 0.7288 / 0.7524 | 0.4501 / 0.5220 | 0.3486 / 0.3735 |
| INT4 | close | 0.8221 / 0.8963 | 0.7119 / 0.7412 | 0.4116 / 0.5186 | 0.3417 / 0.3779 |
| INT8 | open | 0.7029 / 0.7604 | 0.6074 / 0.6287 | 0.3582 / 0.4484 | 0.2886 / 0.3186 |
| INT4 | open | 0.6954 / 0.7458 | 0.5861 / 0.6052 | 0.3517 / 0.4537 | 0.2975 / 0.3316 |

### INT4 minus INT8, percentage points

| scope | page R@5 | page R@10 | page nDCG@5 | page nDCG@10 | layout R@5 | layout R@10 | layout nDCG@5 | layout nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| close | -0.88 | +0.29 | -1.70 | -1.12 | -3.85 | -0.35 | -0.69 | +0.43 |
| open | -0.75 | -1.46 | -2.12 | -2.35 | -0.65 | +0.53 | +0.89 | +1.30 |

## Encoding record

The run used separate normalized 2048-dimensional caches:

| precision | query seconds | page seconds | layout seconds |
| --- | ---: | ---: | ---: |
| INT8 | 6.38 | 608.00 | 2080.79 |
| INT4 | 6.21 | 602.69 | 2143.51 |

These times are operational records only. Because both runs explicitly used
`ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`, they are not a general throughput comparison against
other cache configurations or against another model's benchmark.

## Conclusion

On this 200-query screening subset, Qwen3 INT8 is the more stable quality choice. INT4 is
close on page Recall@10 and some open layout nDCG metrics, but loses close layout Recall@5
by 3.85 percentage points and open page nDCG@5 by 2.12 points. This is a screening result,
not a full-dataset production quality claim.
