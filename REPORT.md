# Qwen3-VL-Embedding-2B OpenVINO Benchmark Report

日期：2026-08-08

本报告记录 Qwen3-VL-Embedding-2B 在 OpenVINO GenAI 2026.3 patched build 上的完整
benchmark 计划、输入定义、输出维度定义和实测结果。原始结果在
[benchmark_results](benchmark_results/)；自动汇总在
[benchmark_results/BENCHMARK_RESULTS.md](benchmark_results/BENCHMARK_RESULTS.md)。

## 1. 测试目标

目标是按现有 `gme-Qwen2-VL-2B-Instruct-Running/08_benchmark_matrix.py` 的思路，
覆盖三类输入和两种测试层级：

1. baseline：FP32/INT8/INT4，CPU/GPU，text/image/fused，确认不同精度和设备的基本
   端到端成本；
2. detailed：GPU INT8/INT4，精确文本长度和图片尺寸，统一使用 native 2048 维输出，
  作为部署选型和服务容量规划依据。

`text` 是纯文本 embedding，`image` 是纯图片 embedding，`fused` 是同一次调用中的
text + image embedding。每次调用返回一个 `[1, D]` 向量。

## 1.1 硬件与软件环境

本 benchmark 的完整环境快照见
[benchmark_results/ENVIRONMENT.json](benchmark_results/ENVIRONMENT.json)。关键硬件和软件如下：

| 项目 | 值 |
|---|---|
| Host/CPU | M2 AI Mini，Intel Core Ultra 7 356H，Panther Lake-P，16 cores / 16 logical CPUs |
| Firmware/board | AMI PB19V1.14，BIOS revision 5.39，PTBNB V1.1 motherboard |
| GPU | Intel Graphics integrated GPU，PCI `0000:00:02.0`，device `0xb0a0`，xe driver |
| Xe/GPU execution | OpenVINO 32 execution units，OpenCL 32 compute units，VTune 32 XVE |
| GPU memory | 30,705,451,008 bytes，约 28.6 GiB（OpenVINO/OpenCL 报告值） |
| System RAM | 32 GB DDR5 SODIMM，Micron，1/2 slots populated，64-bit，1.1 V |
| RAM speed | **5600 MT/s**，configured **5600 MT/s**，已由 `sudo dmidecode` 验证 |
| Swap | 8 GiB |
| OS/kernel | Ubuntu 24.04.4 LTS / Linux 7.0.0-28-generic |
| OpenVINO | 2026.3.0，runtime build `2026.3.0-22452-c2638477a62-qwen-ttft-optimizations` |
| GenAI | 2026.3.0.0 patched Qwen3-VL build |
| Python | 3.12.3，`qwen_env` |
| CPU DMI speed | Max/current 4.7 GHz（DMI）；OS governor snapshot 为 `powersave` |
| Power policy | `intel_pstate` active，governor `powersave` |

benchmark 的 repeat/warmup、精度、设备、文本长度、图像尺寸和 batch 配置也在该 JSON
中保存，结果页顶部同时展示环境摘要。

## 2. 运行前提和修复边界

模型由官方路径导出：

```bash
optimum-cli export openvino \
  --model /home/chaofan/model_benchmark/benchmark/models/Qwen3-VL-Embedding-2B \
  --task feature-extraction \
  --weight-format fp32 \
  /home/chaofan/model_benchmark/benchmark/models/Qwen3-VL-Embedding-2B-FP32
```

INT8/INT4 只改变 `--weight-format`。原始 HF 目录不能直接作为
`EmbeddingPipeline` 的模型目录。

### 2.1 正式修复

OpenVINO GenAI 2026.3 的 Qwen3-VL 无图路径原本创建：

```text
visual_pos_masks        = [batch, 1]
deepstack_visual_embeds = [3, 1, hidden]
```

但 language IR 会在真实 sequence length 上消费这两个辅助输入。修复后按实际输入
长度创建：

```text
visual_pos_masks        = [batch, sequence_length]
deepstack_visual_embeds = [3, sequence_length, hidden]
```

源码位置：[classes.cpp](../benchmark/openvino.genai/src/cpp/src/visual_language/qwen3_vl/classes.cpp)

统一 patch：[0001-fix-qwen3-vl-embedding-2b-2026.3.patch](patches/0001-fix-qwen3-vl-embedding-2b-2026.3.patch)

该 patch 以官方 GenAI 2026.3 release commit
`bd8d6542e3ca1ac30042d5d8d4202ce00b5f4af0`（tag `2026.3.0.0`，
`origin/releases/2026/3`）为基线，必须先在 pristine checkout 应用，
再重新编译和安装。

之前的 `[3] -> [4] position_ids` 实验与 IR 的 `[3, ?, ?]` 输入不匹配，不能作为
修复，已删除，不纳入 benchmark。

### 2.2 Batch 修复

单条 text/image/fused 修复后，首次尝试 `pipeline.embed([text_1, text_2])` 又暴露了
第二个问题：`EmbeddingPipeline` 为每条文本生成 extra inputs，但最后只保留了最后一条
的 `visual_pos_masks` 和 `deepstack_visual_embeds`。batch=2 时 language IR 因此收到
`inputs_embeds` 的 batch=2，却收到未扩展的辅助 tensor，触发 `Reshape_2` shape error。

本次 batch 修复在 `EmbeddingPipeline` infer 前按 batch 扩展：

```text
visual_pos_masks        -> [batch, max_sequence_length]
deepstack_visual_embeds -> [num_layers, batch * visual_sequence, hidden]
```

最终实现按每个 batch item 保存并拼接 extra inputs，因此支持不同文本长度的 ragged
batch，而不是复制最后一条样本的 tensor。

修复后 N=2 的 text/image/fused 均返回 `[2,2048]`，随后才执行完整 N=1/2/4/8/16/32
batch benchmark。相关源码修改位于 `embedding_pipeline.cpp`、`inputs_embedder.*` 和
`qwen3_vl/classes.*`。

## 3. 测量口径

- 计时为 warm end-to-end `EmbeddingPipeline.embed()` 调用，包含 tokenizer、图像
  tensor 处理和 OpenVINO inference，统一返回 native 2048 维；
- 不包含 pipeline 构造、模型读取和首次 compile；`construct_ms` 单独记录；
- 默认探索运行是 `warmup=1, repeat=3`；正式发布数据建议 `warmup=3, repeat=20`；
- `mean_ms` 是 repeat 次调用的算术平均，JSON 同时保存 `stdev_ms`、`min_ms`、
  `max_ms`；
- baseline 默认使用确定性 448x448 RGB 梯度图，image/fused 结果因此可重复；
- detailed image/fused 使用多个确定性 RGB 梯度图，只比较 workload 规模，不将图像
  内容质量误读为性能结论；
- 输出 JSON 是权威原始数据，Markdown 仅是可读汇总。
- batch 额外记录 `ms/call` 和 `ms/embedding = ms/call / N`；batch 输出 shape 应为
  `[N,2048]`。

## 4. 文本长度设计

文本 workload 使用 OpenVINO GenAI `Tokenizer` 逐步生成 ASCII filler，并验证
`input_ids.shape[1]` 精确命中目标。目标长度为：

```text
32, 64, 128, 256, 512, 1024 tokens
```

这里的 token 数是 Qwen3-VL OpenVINO tokenizer 对最终纯文本输入的 token 数，不是
字符数，也不是 Python `len(text)`。每个长度对应的实际字符串、字符数和确认后的
`token_count` 都写入 `detailed.json`。

这样做与 GME benchmark 的“effective token length”思想一致，同时避免把不同 tokenizer
或 prompt 模板的字符长度当成可比变量。

## 5. 图像 workload 设计

详细 image/fused 使用以下尺寸：

| 输入 | 目的 |
|---|---|
| 64x64 | 最小输入/预处理下界探测 |
| 224x224 | 常见小图 |
| 448x448 | baseline 和常见视觉输入基准 |
| 672x672 | 中等图像 |
| 960x480 | 非正方形页面/横向输入 |

Qwen3-VL 的实际 processor 可能按模型的 min/max pixel 规则调整视觉网格，因此
benchmark JSON 保存原始图片路径和最终 timing；当前脚本不把像素数直接宣称为视觉
token 数。若需要视觉 token 精确审计，应从对应 processor 的 debug 输出补充网格信息。

## 6. 输出维度与 latency

Qwen3-VL-Embedding-2B 原生输出维度是 2048。官方模型说明的 MRL/custom dimension
能力意味着前缀子空间，例如 `embedding[:64]`、`embedding[:128]`、`embedding[:256]`
和 `embedding[:512]`，可以单独用于检索；实际部署通常在 native embedding 产生后
截断，并按需要重新 L2-normalize。

`EmbeddingPipeline` 2026.3 没有原生 `output_dim` 参数，且主干 IR 仍按 hidden size
2048 执行。因此选择 64、128、256、512 或 1024 维时，通常仍会完整执行一次 2048
维的 language/vision inference，只是在最后复制/截断向量。低维选择的收益是：

- 向量存储和索引内存下降；
- 相似度计算和内存带宽下降；
- 可能改善下游索引吞吐。

它通常不会降低本 benchmark 所测的端到端模型 inference latency。除非后续显式加入
模型图中的低维 projection、量化或硬件专用 output head，否则为 64/512/1024 等维度
分别重复 latency 测试会产生误导。因此本报告所有性能结果统一记录 native `[1, 2048]`；
低维 MRL 只需要做检索质量、索引大小和下游吞吐评估，不再做低维 latency 矩阵。

## 7. 测试矩阵

### 7.0 Baseline 的精确输入

当前结果页中的 baseline 使用以下固定 workload：

```text
text = "a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a"
tokenizer token_count = 32
image = benchmark_results/assets/gradient_448x448.png
image shape = [448, 448, 3]
image dtype = uint8 RGB
```

三个模式的实际调用分别是：

```python
pipeline.embed(text)
pipeline.embed(images=[image_tensor])
pipeline.embed(text, images=[image_tensor])
```

因此 baseline 表中的 `Text`、`Image`、`Fused` 分别对应纯文本、纯图片和同一调用中
同时传入该 32-token 文本及该 448x448 图片。模型构造/编译时间单独列在 `Construct`，
不计入三个 warm latency 数字。

### 7.1 Baseline

| 维度 | 覆盖范围 |
|---|---|
| precision | FP32、INT8、INT4 |
| device | CPU、GPU；不可用设备记录为 unavailable，不静默删除 |
| mode | text、image、fused |
| output | native 2048 |
| image | 448x448 |
| text | 32 tokenizer tokens |

理论上最多 18 个结果单元：3 precision x 2 device x 3 mode。每个单元都记录
construct time 和 warm timing statistics。

### 7.2 Detailed text

| 维度 | 覆盖范围 |
|---|---|
| precision | INT8、INT4 |
| device | GPU |
| token length | 32、64、128、256、512、1024 |
| output dimension | native 2048 |
| mode | text |

理论上为 2 x 6 = 12 个 text 单元。

### 7.3 Detailed image/fused

| 维度 | 覆盖范围 |
|---|---|
| precision | INT8、INT4 |
| device | GPU |
| image sizes | 64x64、224x224、448x448、672x672、960x480 |
| output | native 2048 |
| mode | image、fused |

理论上为 2 x 5 x 2 = 20 个 image/fused 单元。

### 7.4 Batch

batch 使用 GPU INT8/INT4，N 为 `1, 2, 4, 8, 16, 32`。每一行都是一次公开 API 调用，
不是 N 次独立调用：

```python
pipeline.embed([text] * N)
pipeline.embed([""] * N, images=[image_tensor])
pipeline.embed([text] * N, images=[image_tensor])
```

其中 image/fused 采用一张共享的 448x448 图片和 N 条文本行，因为当前
`EmbeddingPipeline` 的 multimodal API 接收的是一组 images 与一组 texts，而不是
逐行的 `(text_i, image_i)` pairs。结果同时记录 `ms/call`、`ms/embedding` 和返回
shape `[N,2048]`。

理论上为 2 x 6 x 3 = 36 个 batch 单元，原始结果在
[batch.json](benchmark_results/batch.json)。

## 8. 复现命令

先构建并安装 patched GenAI：

```bash
source /home/chaofan/envs/qwen_env/bin/activate
cmake --build /home/chaofan/model_benchmark/benchmark/openvino.genai/build -j2
cmake --install /home/chaofan/model_benchmark/benchmark/openvino.genai/build \
  --config Release \
  --prefix /home/chaofan/model_benchmark/benchmark/openvino/install
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
cd /home/chaofan/model_benchmark/Qwen3-VL-Embedding-2B-Running
```

探索运行：

```bash
python benchmark_qwen3_vl.py --suite baseline --repeat 3 --warmup 1
python benchmark_qwen3_vl.py --suite detailed --repeat 3 --warmup 1
python benchmark_qwen3_vl.py --suite batch --repeat 3 --warmup 1
```

正式运行：

```bash
python benchmark_qwen3_vl.py --suite all --repeat 20 --warmup 3
```

单独指定页面图片做 baseline：

```bash
python benchmark_qwen3_vl.py --suite baseline \
  --image /path/to/page.png --repeat 20 --warmup 3
```

## 9. 实测结果

完整数字由脚本写入：

- [baseline.json](benchmark_results/baseline.json)
- [detailed.json](benchmark_results/detailed.json)
- [batch.json](benchmark_results/batch.json)
- [BENCHMARK_RESULTS.md](benchmark_results/BENCHMARK_RESULTS.md)

其中 `BENCHMARK_RESULTS.md` 包含 baseline 表、每个文本长度的 native 2048 维表、
以及每个图片尺寸的 image/fused native 2048 维表和 batch N 扩展表。这里不手工复制 JSON 数字，避免
重新运行后报告与原始结果漂移。

当前 batch 结果使用最终通用 batch 修复、GPU、`repeat=20`、`warmup=3`，覆盖
INT8/INT4 和 N=1/2/4/8/16/32；每个 N 的 text/image/fused 输出 shape 分别为
`[N,2048]`。baseline 和 detailed 结果同样保留在对应 JSON 中，当前汇总页是三套结果
的合并视图。

### 结果阅读规则

- baseline 先回答哪种 precision/device/mode 最快；
- detailed text 观察 token length 对 latency 的增长趋势；
- detailed image/fused 观察图像规模和 fused 额外文本的成本；
- 输出统一为 native 2048；MRL 低维只做质量/索引存储评估，不作为 IR 推理降维加速承诺；
- CPU/GPU 跨设备比较必须保留同一 workload、同一 repeat/warmup 和同一电源状态；
- 生产引用应优先使用 `repeat=20` 或更高的正式结果，不引用探索运行。

## 10. PDF 文档 Embedding 实测

已使用本目录的 `060_OT_Downlook_PR_Error.pdf` 完成与 GME 文档评测对应的 GPU INT8/INT4
测试。完整方法和原始数据见：

- [document benchmark README](document_embedding_eval_int8_int4_gpu/README.md)
- [document benchmark JSON](document_embedding_eval_int8_int4_gpu/results/benchmark.json)
- [document benchmark report](document_embedding_eval_int8_int4_gpu/results/REPORT.md)

本次输入为 14 页 PDF，页面以 150 DPI 渲染。每种 precision 测试 `page_text`、
`text_chunk_256`、`page_image_150`、`page_fused_150`，batch 档位为 `1/4/8/14`，
每个单元 repeat=3。结果全部为 native 2048 维。

关键结果：

| Precision | 表示 | batch=4 mean | items/s | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| INT8 | page_text | 539.7 ms | 25.94 | 0.125 | 0.625 | 0.625 | 0.385 |
| INT8 | text_chunk_256 | 525.7 ms | 26.63 | 0.125 | 0.625 | 0.625 | 0.385 |
| INT8 | page_image_150 | 50437.5 ms | 0.28 | 0.250 | 0.750 | 0.875 | 0.507 |
| INT8 | page_fused_150 | 51233.5 ms | 0.27 | 0.125 | 0.625 | 0.625 | 0.406 |
| INT4 | page_text | 514.4 ms | 27.21 | 0.125 | 0.500 | 0.750 | 0.380 |
| INT4 | text_chunk_256 | 499.9 ms | 28.00 | 0.125 | 0.500 | 0.750 | 0.380 |
| INT4 | page_image_150 | 50463.6 ms | 0.28 | 0.250 | 0.750 | 0.750 | 0.497 |
| INT4 | page_fused_150 | 51105.7 ms | 0.27 | 0.250 | 0.500 | 0.750 | 0.461 |

Qwen3 的 multimodal API 不支持 GME 风格的一次调用内逐行 `(text_i, image_i)` pair，
所以 image/fused 的 batch 档位采用逐页配对调用分组计时，避免把全部页面图片错误地
附加到每条文本。该差异和完整解释见文档 benchmark README。

## 11. 质量与部署前检查

## 12. PA/SDPA backend 结论

Qwen3-VL-Embedding-2B 不使用 Qwen3.5 生成式 `VLMPipeline` 的 PA/SDPA backend switch。
它由 `EmbeddingPipeline` 直接编译 stateless language/vision embedding IR；当前
language/vision graph 含 `ScaledDotProductAttention`，不含 `PagedAttentionExtension`。
完整源码调用链、官方默认 PA 的适用范围和为什么不补跑 PA/SDPA A/B 见
[GME_Qwen3VL_Embedding_PA_SDPA_调查.md](../benchmark/tests/GME_Qwen3VL_Embedding_PA_SDPA_调查.md)。

本 benchmark 只验证可运行性和性能，不替代以下检查：

1. native 2048 维与原始 Transformers embedding 的 cosine parity；
2. 64/128/256/512/1024 MRL 前缀向量的检索质量回归；
3. PDF page image 的真实预处理和 fused 质量；
4. 下游 index 的维度、归一化、序列化和迁移；
5. INT8/INT4 在真实文档集合上的 recall/NDCG；
6. 长文本 1024 token 和更长输入的内存峰值。

当前建议以 INT8 GPU 作为第一部署基线，INT4 GPU 作为容量/速度候选。FP32 CPU/GPU
baseline 用于 correctness 和设备回归，不建议直接据此选择生产精度。
