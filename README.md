# Qwen3-VL-Embedding-2B OpenVINO 2026.3

本目录包含 Qwen3-VL-Embedding-2B 的 OpenVINO GenAI 修复 patch、可复现 benchmark
脚本和测试结果。模型目录必须是 `feature-extraction` 导出的 OpenVINO 目录，不能把
原始 Hugging Face 权重目录直接传给 `EmbeddingPipeline`。

## 先决条件

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
cd /home/chaofan/model_benchmark/Qwen3-VL-Embedding-2B-Running
```

`benchmark/openvino/install` 必须来自包含 Qwen3-VL 修复的 GenAI build：

```bash
cmake --build /home/chaofan/model_benchmark/benchmark/openvino.genai/build -j2
cmake --install /home/chaofan/model_benchmark/benchmark/openvino.genai/build \
  --config Release \
  --prefix /home/chaofan/model_benchmark/benchmark/openvino/install
```

统一修复 patch：[0001-fix-qwen3-vl-embedding-2b-2026.3.patch](patches/0001-fix-qwen3-vl-embedding-2b-2026.3.patch)。

该 patch 以官方 OpenVINO GenAI 2026.3 release commit
`bd8d6542e3ca1ac30042d5d8d4202ce00b5f4af0`（tag `2026.3.0.0`，
`origin/releases/2026/3`）为基线，包含纯文本 shape 修复和
multimodal batch 修复。复现方式：

```bash
cd /path/to/openvino.genai
git checkout bd8d6542e3ca1ac30042d5d8d4202ce00b5f4af0
git apply /path/to/Qwen3-VL-Embedding-2B-Running/patches/0001-fix-qwen3-vl-embedding-2b-2026.3.patch
cmake --build build -j2
```

## Benchmark

脚本：[benchmark_qwen3_vl.py](benchmark_qwen3_vl.py)

完整测试包含：

- baseline：FP32/INT8/INT4 x CPU/GPU x text/image/fused，原生 2048 维输出；
- detailed text：GPU INT8/INT4，精确 32/64/128/256/512/1024 tokenizer tokens；
- detailed image/fused：GPU INT8/INT4，64x64、224x224、448x448、672x672、960x480；
- batch：GPU INT8/INT4，N=1/2/4/8/16/32，text/image/fused 的单次批量调用；
- detailed output：所有性能结果统一使用 native 2048 维输出；MRL 低维向量只作为质量/存储后处理，不单独测 latency。

OpenVINO GenAI 的 `EmbeddingPipeline` 原生返回 2048 维。MRL 允许使用
`embedding[:64]`、`embedding[:128]`、`embedding[:256]` 等前缀子空间，但这属于
推理后的截断/归一化，不会让主干 language/vision IR 只计算低维向量。因此 benchmark
只把 native 2048 维作为性能口径；低维 MRL 只需另做检索质量和索引大小评估。

快速 smoke / 低成本探索：

```bash
python benchmark_qwen3_vl.py --suite baseline --repeat 3 --warmup 1
python benchmark_qwen3_vl.py --suite detailed --repeat 3 --warmup 1
python benchmark_qwen3_vl.py --suite batch --repeat 3 --warmup 1
```

正式结果建议使用至少 20 次 warm call：

```bash
python benchmark_qwen3_vl.py --suite all --repeat 20 --warmup 3
```

输出目录默认为 `benchmark_results/`：

- `baseline.json`：全精度、CPU/GPU 基线原始结果；
- `detailed.json`：长度、图像尺寸、输出维度的完整矩阵；
- `batch.json`：GPU INT8/INT4 的 N 扩展结果，包含 `ms/call` 和 `ms/embedding`；
- `BENCHMARK_RESULTS.md`：自动生成的可读表格；
- `manifest.json`：测试参数和维度语义；
- `assets/`：确定性梯度图 workload。

## 文档 Embedding Benchmark

已加入与 GME `document_embedding_eval_int8_int4_gpu` 对齐的 PDF 文档评测：

[document_embedding_eval_int8_int4_gpu](document_embedding_eval_int8_int4_gpu/)

测试输入是本目录的 [060_OT_Downlook_PR_Error.pdf](060_OT_Downlook_PR_Error.pdf)，覆盖：

- GPU INT8/INT4；
- PDF 原生文本 page embedding；
- 256 tokenizer tokens、64 tokens overlap 的 text chunks；
- 150 DPI page image embedding；
- page text + 对应 page image 的 fused embedding；
- batch `1/4/8/14`；
- items/s、tokens/s、模型加载时间、RSS、Recall@1/3/5、MRR 和跨表示 cosine。

运行：

```bash
cd document_embedding_eval_int8_int4_gpu
python run_eval.py
```

完整原始结果：[results/benchmark.json](document_embedding_eval_int8_int4_gpu/results/benchmark.json)

可读报告：[results/REPORT.md](document_embedding_eval_int8_int4_gpu/results/REPORT.md)

Qwen3 multimodal API 不支持 GME 风格的 `(text_i, image_i)` 成对 batch，因此 image/fused
保持逐页配对调用，并以 batch size 作为分组档位；具体口径见该目录的 README。

## 结果与方法

详细测试计划、计时口径、结果解读和生产迁移注意事项见
[REPORT.md](REPORT.md)。实际数字以 `benchmark_results/*.json` 和
[benchmark_results/BENCHMARK_RESULTS.md](benchmark_results/BENCHMARK_RESULTS.md) 为准。

## 清理说明

旧的 position-id `[3] -> [4]` 实验没有修复问题，已删除。官方 release 的支持边界、
失败原因和修复过程见 [HISTORICAL_DIAGNOSIS.md](HISTORICAL_DIAGNOSIS.md)。旧的诊断 JSON、fresh export
日志、临时 smoke-test 脚本和临时安装目录也已清理；新的 benchmark JSON 是唯一性能结果来源。
