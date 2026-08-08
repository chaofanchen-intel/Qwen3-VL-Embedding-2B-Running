# Qwen3-VL-Embedding-2B 文档 Embedding 评测

本目录复用 GME 的 `document_embedding_eval_int8_int4_gpu` 评测思路，测试
`060_OT_Downlook_PR_Error.pdf` 在 patched OpenVINO GenAI 2026.3 + Intel GPU 上的
文档 embedding 性能和简单检索质量。

## 测试范围

- device：`GPU`
- precision：`INT8`、`INT4`
- PDF：`../060_OT_Downlook_PR_Error.pdf`
- page image：PDF 原生页面以 150 DPI 渲染为 PNG
- native output：`[N, 2048]`
- batch：`1, 4, 8, 14`
- repeat：每个性能单元 3 次，包含 1 次 warm-up call，JSON 保存 mean/p50/p95/min/max

## 表示方案

| 表示 | 输入 | 用途 |
|---|---|---|
| `page_text` | 每页 PDF 原生文本层 | 文本页面基线 |
| `text_chunk_256` | 每页文本按 256 tokenizer tokens、64 tokens overlap 切块 | 长文本切分和细粒度检索 |
| `page_image_150` | 每页 150 DPI RGB PNG | 页面视觉 embedding |
| `page_fused_150` | 一页的文本与对应页面图像 | 图文联合 embedding |

文本使用 `embedding_prompt="Represent the user's input."`；质量 query 也使用相同
prompt，以保持 Qwen3 模型的 instruction-aware 输入语义一致。实际页面 token 数、chunk
数量、图片尺寸和文件大小都写入 `results/benchmark.json`。

## Qwen3 batch API 注意事项

Qwen3 `EmbeddingPipeline` 的 multimodal Python API 接收的是 `texts` 集合和 `images`
集合，不提供 GME 风格的逐行 `(text_i, image_i)` pair batch 参数。若把 14 张图片和
14 条文本同时传入，模型会把图片集合附加到每条文本，不能用来表达“一行一页”。因此：

- `page_text` 和 `text_chunk_256` 使用真正的 text batch，一次调用返回 `[N,2048]`；
- `page_image_150` 和 `page_fused_150` 保持严格的 page pair 语义，每个页面独立调用，
  以 batch size 作为分组/调度档位，结果记录整组 wall time、items/s 和 `[items,2048]`
  输出；
- 这两类 image/fused 的 `batch_size` 不是一次传入 N 张图片，而是 paired page 调用
  的 chunk size。这样不会产生跨页面的错误融合，且仍然能与 GME 的 batch 档位对照。

## 质量测试

脚本内置 8 个工程 query 及人工标注相关页：

- halo effect / edge bump
- Downlook PR error
- flux coverage
- PR learning substrate
- threshold and area
- lighting adjustment
- Vision inspection log
- Vision Dump / system files

对四种表示计算：

- Recall@1、Recall@3、Recall@5
- MRR
- 每个 query 的 top-5 页面和分数
- 对应页面的 text/image/fused 跨表示 cosine（观察指标，不假设不同模态必然等价）

这些指标是单个 14 页工程 PDF 的诊断结果，不应替代真实文档集合上的 recall/NDCG
回归。文本层优先使用 PDF 原生文字；没有引入 OCR，避免把 OCR 误差混入本次结果。

## 运行

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
cd /home/chaofan/model_benchmark/Qwen3-VL-Embedding-2B-Running/document_embedding_eval_int8_int4_gpu
python run_eval.py
```

输出：

- `results/benchmark.json`：PDF、预处理、模型加载、性能、质量的完整原始结果；
- `results/REPORT.md`：自动生成的可读摘要；
- `pages/`：150 DPI 中间页面图像。

本 benchmark 需要使用包含统一 Qwen3 修复 patch 的 GenAI build。patch 及其官方
2026.3 release 基线见上级目录 [README.md](../README.md) 和
[patches/0001-fix-qwen3-vl-embedding-2b-2026.3.patch](../patches/0001-fix-qwen3-vl-embedding-2b-2026.3.patch)。
