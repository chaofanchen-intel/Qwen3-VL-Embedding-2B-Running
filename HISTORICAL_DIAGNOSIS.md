# Qwen3-VL-Embedding-2B 历史诊断与模型选型归档

日期：2026-08-08

本文保留在 benchmark 重构前的模型选型、官方代码覆盖情况、失败证据和修复过程。
性能新结果不写在本文，统一见 [REPORT.md](REPORT.md) 和
[benchmark_results/BENCHMARK_RESULTS.md](benchmark_results/BENCHMARK_RESULTS.md)。

## 1. 选型结论

`Qwen3-VL-Embedding-2B` 是 `gme-Qwen2-VL-2B-Instruct` 最接近的升级方向：两者都
支持 text、image、text+image 单向量 embedding；Qwen3-VL-Embedding 还支持 screenshot、
video、32K context、30+ languages、instruction-aware embedding 和 MRL 可变维度。
Qwen3-VL-Embedding-2B 的 native 输出是 2048 维，旧 GME 是 1536 维，因此迁移会影响
下游 index/schema。

候选取舍记录：

| 模型 | 与 GME 功能等价 | OpenVINO 2026.3 路径 | 判断 |
|---|---|---|---|
| Qwen3-VL-Embedding-2B | 是 | `EmbeddingPipeline` | 长期首选，需使用 patched GenAI |
| Qwen3-VL-Embedding-8B | 是 | 同系列但转换和显存压力更大 | 质量上限研究，不是近期部署首选 |
| Qwen3-Embedding-0.6B | 否，仅文本 | 文本 embedding 路径 | 纯文本高速备选 |
| Jina Embeddings v4 | 是，面向视觉文档 | 没有本机确认的原生路径 | 需要自定义导出 |
| ColQwen2.5 | 多向量，不是同一 API | 需要 ColBERT multi-vector 流程 | 不作为 drop-in 替代 |
| BGE-VL-MLLM-S2 | 大体相近 | 需要 custom code | 不建议近期迁移 |

## 2. 官方支持代码中的覆盖和失败标记

本机 OpenVINO GenAI 2026.3 checkout：

```text
/home/chaofan/model_benchmark/benchmark/openvino.genai
```

官方测试文件：

```text
benchmark/openvino.genai/tests/python_tests/test_rag.py
```

关键位置：

```text
line 33-36: MULTIMODAL_EMBEDDINGS_TEST_MODELS
line 34:     # "Qwen/Qwen3-VL-Embedding-2B",
line 35:     "optimum-intel-internal-testing/tiny-random-qwen3-vl-embedding"

line 352: @pytest.mark.xfail(reason="Ticket - CVS-189808")
line 368: @pytest.mark.xfail(reason="Ticket - CVS-189808")
line 393: @pytest.mark.xfail(reason="Ticket - CVS-189808")
```

352--365 是真实 image+text 与 HF cosine 对照；368--390 是 Sentence Transformers
对照；393 开始是 Sentence Transformers 与 Transformers 一致性对照。也就是说，
2026.3 已经有 `EmbeddingPipeline` API，但真实 `Qwen/Qwen3-VL-Embedding-2B` 在这份
release test model 列表中被注释，相关真实多模态一致性用例仍标记为 `CVS-189808`
xfail。这是官方覆盖状态，不代表模型权重本身损坏。

官方导出任务是：

```bash
optimum-cli export openvino \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --task feature-extraction \
  Qwen3-VL-Embedding-2B
```

本机 FP32/INT8/INT4 都按 `feature-extraction` 导出，已经排除 generation task 误用。

## 3. 修复前失败证据

本地模型目录：

```text
benchmark/models/Qwen3-VL-Embedding-2B-FP32
benchmark/models/Qwen3-VL-Embedding-2B-INT8
benchmark/models/Qwen3-VL-Embedding-2B-INT4
```

未修复 GenAI 2026.3 的调用：

```python
pipeline = openvino_genai.EmbeddingPipeline(model_dir, "GPU")
pipeline.embed("What causes the halo effect on the edge bump?")
```

失败节点和错误：

```text
__module.language_model/aten::reshape/Reshape_2
Shape inference input shapes {[1],[3]}
Requested output shape [1,12,1] is incompatible with input shape
```

修复前矩阵：

| IR | text | image | text + image |
|---|---|---|---|
| FP32 | FAIL | OK `[1, 2048]` | OK `[1, 2048]` |
| INT8 | FAIL | OK `[1, 2048]` | OK `[1, 2048]` |
| INT4 | FAIL | OK `[1, 2048]` | OK `[1, 2048]` |

失败不是 INT8/INT4 独有：FP32 也失败；image-only/fused 成功说明问题集中在纯文本
分支。

## 4. 导出和权重对照

为了排除旧导出目录污染，重新使用相同本地 HF source model 做 fresh FP32 export，
仍然在未修复 runtime 上得到同一个 `Reshape_2` 错误。

同一 source model 用模型自带 `scripts/qwen3_vl_embedding.py` 的 Transformers 路径
在 CPU 上成功：

```text
Qwen3VLEmbedder.process([{"text": "What causes the halo effect on the edge bump?"}])
output shape = torch.Size([1, 2048])
```

因此当时的责任边界是：

1. 不是模型权重损坏；
2. 不是 INT8/INT4 量化单独造成；
3. 不是 generation task 导出错误；
4. 是 Qwen3-VL language IR 与 GenAI 纯文本输入准备之间的 shape contract 问题；
5. 官方真实模型测试覆盖仍有限，并且相关一致性用例带有 `CVS-189808` xfail。

## 5. 正式修复

文件：

```text
benchmark/openvino.genai/src/cpp/src/visual_language/qwen3_vl/classes.cpp
```

无图文本路径原本分配：

```text
visual_pos_masks        = [batch, 1]
deepstack_visual_embeds = [3, 1, hidden]
```

改为按照 `input_ids.get_shape()[1]` 分配：

```text
visual_pos_masks        = [batch, sequence_length]
deepstack_visual_embeds = [3, sequence_length, hidden]
```

统一正式 patch：

[0001-fix-qwen3-vl-embedding-2b-2026.3.patch](patches/0001-fix-qwen3-vl-embedding-2b-2026.3.patch)

该 patch 已在 pristine GenAI 2026.3 worktree 上通过 `git apply --check`。之前临时把
`position_ids` 从 `[3, 1, seq]` 改成 `[4, 1, seq]` 的实验失败，因为 IR 顶层输入是
`[3, ?, ?]`；该实验不是修复，已从工作目录删除。

## 6. 修复后的 smoke 结果

在 patched GenAI build 上，FP32/INT8/INT4 的 text、image、fused 均返回 `[1, 2048]`。
早期 INT8/INT4 GPU smoke 记录如下，时间只是单次 wall time，不是正式 benchmark：

| Precision | text | image | fused |
|---|---:|---:|---:|
| INT8 | 约 171--172 ms | 约 4.0--4.1 s | 约 3.6--3.7 s |
| INT4 | 约 235--250 ms | 约 4.0--4.1 s | 约 3.6--3.7 s |

后续正式性能数字不在本文维护，避免与新的 workload、repeat 和电源状态混用。

## 7. 历史结论的边界

“未修复官方 2026.3 build 的 text 路径 failed”和“patched GenAI build 的三路径
通过”是两个不同结论，不能混写：

- 原版 release path：纯文本失败，官方真实模型覆盖也不完整；
- 本地 patched path：FP32/INT8/INT4 的 text/image/fused 可运行；
- 是否可替换生产 GME：还要看 native 2048 维 cosine parity、INT8/INT4 质量、PDF
  page benchmark、下游 index 迁移和 MRL 低维质量。

## 8. 为什么官方宣称支持，但原始 2026.3 仍不能直接跑

这里的“支持”需要拆成三个层次，不能把它理解成“所有真实模型、所有输入组合和所有
batch shape 已经通过 release validation”：

1. **API 支持**：2026.3 已提供 `EmbeddingPipeline`，并有 multimodal embedding
  文档、模型导出命令和 Qwen3-VL 相关代码路径；因此官方可以宣称该模型系列进入
  GenAI 支持范围。
2. **测试覆盖**：本地 2026.3 `tests/python_tests/test_rag.py` 的真实模型列表中，
  `Qwen/Qwen3-VL-Embedding-2B` 仍被注释，使用的是 tiny random model；真实多模态
  对照测试在 352、368、393 行附近标记 `xfail(reason="Ticket - CVS-189808")`。
  这说明 release 已有接口和测试骨架，但真实模型的完整一致性验证仍未闭合。
3. **本地模型图契约**：本机按照官方 `--task feature-extraction` 重新导出的 FP32、
  INT8、INT4 IR 都能被 pipeline 构造，但真实 infer 暴露了 exporter 图与 C++ 输入
  准备代码之间的 shape contract 缺口。模型文件不是“不能加载”，而是某些合法输入
  路径在执行时使用了不匹配的辅助 tensor shape。

### 8.1 原始 text 失败的具体原因

Qwen3-VL language IR 的关键输入包括：

```text
position_ids             [3, ?, ?]
inputs_embeds            [?, ?, 2048]
visual_pos_masks         [?, ?]
deepstack_visual_embeds  [?, ?, 2048]
```

原始 GenAI 2026.3 `qwen3_vl/classes.cpp` 在无 image/video 的 text 路径中分配：

```text
visual_pos_masks        [batch, 1]
deepstack_visual_embeds [3, 1, hidden]
```

但导出的 language graph 的内部 reshape 会按照实际 sequence length 消费这些辅助
输入。于是单条 text infer 在 `__module.language_model/aten::reshape/Reshape_2` 失败：

```text
Shape inference input shapes {[1],[3]}
Requested output shape [1,12,1] is incompatible with input shape
```

正确修复是使用 `input_ids.shape[1]`：

```text
visual_pos_masks        [batch, sequence_length]
deepstack_visual_embeds [3, sequence_length, hidden]
```

### 8.2 原始 batch 失败的具体原因

应用单条 text 修复后，`EmbeddingPipeline.embed([text_1, text_2])` 又暴露了另一层
问题。`EmbeddingPipeline::multimodal_embed()` 会逐条调用 inputs embedder，但原始代码
在 infer 前只读取一次 `m_inputs_embedder->get_lm_extra_inputs()`。因此 batch 内每个 item
自己的 Qwen3 extra inputs 没有被保存和拼接，batch=2 时会出现：

```text
Requested output shape [2,3,1] is incompatible with input shape
```

统一 patch 现在会：

- 在每个 `BatchItem` 中保存该 item 的 extra inputs；
- 将 `visual_pos_masks` 按 `[batch, max_sequence_length]` 拼接，并保留每条样本自己的 mask；
- 将 `deepstack_visual_embeds` 按 layer 维度拼接每条样本自己的 visual sequence；
- 对 ragged text batch 使用真实 sequence length，而不是复制最后一条样本的 tensor。

因此修复覆盖的是 text-only 单条路径、multimodal 单条路径保持兼容，以及 text/image/fused
batch 路径。

## 9. Batch 发现

单条输入修复后，首次测试 `EmbeddingPipeline.embed([text_1, text_2])` 仍然失败：

```text
Shape inference input shapes {[3],[3]}
Requested output shape [2,3,1] is incompatible with input shape
```

原因是每条文本的 `visual_pos_masks`/`deepstack_visual_embeds` 没有随 batch 一起扩展。
这已在统一 patched GenAI source 中修复，N=2、ragged text N=2、N=32 的 text/image/fused
均已验证返回对应的 `[N,2048]`。
正式 batch 结果不属于原始 release 的历史 smoke 结果，统一记录在
[benchmark_results/batch.json](benchmark_results/batch.json)。
