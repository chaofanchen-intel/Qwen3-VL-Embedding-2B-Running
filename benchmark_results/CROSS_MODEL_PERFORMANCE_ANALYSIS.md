# GME-Qwen2-VL-2B-Instruct vs Qwen3-VL-Embedding-2B
# 1024-token 性能差异分析

日期：2026-08-11

## 1. 结论

Qwen3-VL-Embedding-2B 在 1024-token text 上比 GME Python 慢，并不是单纯的
GPU boost/throttle 测量假象；在严格交替、相同进程生命周期、相同 warmup/repeat 的
per-call 复测中，差距仍然存在：

```text
GME INT4 median 约 280.5 ms
Qwen3 INT4 median 约 356.0 ms
Qwen3 / GME 约 1.269x
```

但是此前“Qwen3 在 32/128 token 都比 GME 快”的前提不完全成立：

- 32 token：Qwen3 更快；
- 128 token：严格交替复测中 Qwen3 已比 GME 慢；
- 512 token：Qwen3 比 GME 慢约 14.5%；
- 1024 token：Qwen3 比 GME 慢约 26.9%。

所以真实趋势是：**Qwen3 只有极短输入在本机更快；序列变长后，Qwen3 的模型计算结构使 latency 逐步高于 GME。**

## 2. 严格交替复测

测试条件：

- Intel Core Ultra 7 356H iGPU；
- GPU INT4；
- GME `gme_openvino.py`，`streams=1`、`batch_mode=padded`；
- Qwen3 `EmbeddingPipeline`；
- 每个模型/长度 3 轮交替；
- 每轮 warmup=14，timed repeat=10；
- N=1，text-only；
- 两边各自 tokenizer 构造目标 workload；
- 结果使用全部 per-call samples 的 median/p10/p90。

| Effective target | GME median | GME p10-p90 | Qwen3 median | Qwen3 p10-p90 | Qwen3/GME |
|---:|---:|---:|---:|---:|---:|
| 32 | 32.72 ms | 32.42-181.60 ms | 28.76 ms | 28.49-29.76 ms | 0.879x |
| 128 | 46.25 ms | 45.96-47.30 ms | 58.87 ms | 58.45-59.18 ms | 1.273x |
| 512 | 146.11 ms | 145.17-146.85 ms | 167.31 ms | 166.04-168.00 ms | 1.145x |
| 1024 | 280.49 ms | 279.07-296.33 ms | 356.00 ms | 354.42-357.45 ms | 1.269x |

32-token GME 的 p90 被一条 181.6 ms outlier 拉高，median 和其余样本仍在约 32.5 ms。
Qwen3 的 per-call 分布在这轮测试中更稳定。

原始交替 samples：

```text
gme-qwen2-vl-int4-hotspot-optimization/results/cross_model_per_call/
```

## 3. Token 长度不是完全相同的输入定义

### GME

GME benchmark 的 `sequence_tokens` 是 **GME prompt 模板展开后的最终有效 token 数**：

```text
system/user/assistant template
+ endoftext
+ user filler
= target sequence_tokens
```

因此 GME 的 1024 表示 language IR 最终收到约 1024 个有效 token。

### Qwen3

Qwen3 多长度 harness 先用 OpenVINO tokenizer 构造 raw text，使：

```text
Tokenizer.encode(raw_text).shape[1] = target
```

但 `EmbeddingPipeline` 的 `format_prompt()` 还会执行 `append_added_special_tokens()`。
对当前 tokenizer，raw text 的默认 encode 已包含 EOS，而 no-special encode 少一个 token；
GenAI 又会把 added special token 追加到 prompt，再由 language tokenizer 编码。
对 1024 workload，tokenizer 模拟结果为：

```text
raw encode:       1024 tokens
no-special text:  1023 tokens
added EOS:        1 token
final re-encode:  1025 tokens
```

因此 Qwen3 的“1024”与 GME 的“1024”不是完全相同的最终 language sequence，Qwen3
大约多 1 token。这个差异很小，不能解释 26.9% latency 差距，但必须记录，避免把两个
数字误称为严格相同的 decoder shape。

## 4. 模型结构差异

两者都是 28 层、约 2B 级别的视觉语言 embedding 模型，但 decoder 宽度和 attention
配置不同。

| 配置 | GME-Qwen2-VL-2B | Qwen3-VL-Embedding-2B |
|---|---:|---:|
| hidden size | 1536 | 2048 |
| intermediate size | 8960 | 6144 |
| decoder layers | 28 | 28 |
| query attention heads | 12 | 16 |
| KV heads | 2 | 8 |
| head dimension | 128 | 128 |
| output dimension | 1536 | 2048 |

### 4.1 Attention projection规模

忽略 bias，单层 Q/K/V/O projection 的权重元素数近似为：

```text
GME:
  Q: 1536*1536
  K: 1536*(2*128)
  V: 1536*(2*128)
  O: 1536*1536
  total ~= 5.50M weights/layer

Qwen3:
  Q: 2048*2048
  K: 2048*(8*128)
  V: 2048*(8*128)
  O: 2048*2048
  total ~= 12.58M weights/layer
```

Qwen3 每层 attention projection 权重约为 GME 的 **2.29x**。在长序列中，attention
projection 和 attention score/value 计算的 token 线性/二次部分会占据越来越大比例，
因此 Qwen3 随序列增长更快是符合结构的。

### 4.2 MLP 不是唯一决定因素

两者的 MLP intermediate size 不同，GME 的 intermediate size 反而更大；不能简单用
“Qwen3 hidden 更大，所以所有层都同比更慢”解释。粗略按 gated MLP 的三矩阵计算量：

```text
GME:   3 * 1536 * 8960 ~= 41.27M weights/layer
Qwen3: 3 * 2048 * 6144 ~= 37.75M weights/layer
```

Qwen3 的 MLP 规模约为 GME 的 0.91x，而 attention projection 明显更大。因此差异
主要不是 MLP 单项，而是 attention projection/head 配置、额外 Q/K/V 工作和整体 IR
执行路径的组合。

### 4.3 GQA 配置不同

GME 只有 2 个 KV heads，Qwen3 有 8 个 KV heads。虽然 Qwen3 的 query head 数为 16、
GME 为 12，Qwen3 的 attention head 数更多；同时 GME 的 KV 压缩更激进，K/V projection
和 KV 相关内存流量更小。长文本时这种差异会比 32-token 下的固定调度开销更明显。

## 5. IR 数据流差异

### GME language path

```text
input_ids [B,S]
  -> token embedding Gather
  -> Qwen2 decoder
  -> last valid token pooling + normalize
  -> [B,1536]
```

GME language IR 直接接收 `input_ids`、`attention_mask` 和空/视觉
`image_embeds[?,1536]`，token embedding 是同一张 language graph 的一部分。

### Qwen3 language path

```text
input_ids [B,S]
  -> separate openvino_text_embeddings_model.xml
  -> inputs_embeds [B,S,2048]
  -> Qwen3 language decoder
  -> postprocessing last-token Gather
  -> NormalizeL2
  -> [B,2048]
```

Qwen3 的 `EmbeddingPipeline` 在 language decoder 前额外调用独立的 text embedding IR，
然后将 `[B,S,2048]` 作为 `inputs_embeds` 输入 language graph。它还需要构造：

```text
position_ids [3,B,S]
visual_pos_masks [B,S]
deepstack_visual_embeds [3,S,2048]
```

这些额外输入和 2048 hidden state copy/postprocessing 会增加固定和线性开销。短文本时
固定开销、kernel launch 和 CPU/GPU 调度占主导；Qwen3 当前实现的短文本调度可能比 GME
更快。长文本时 decoder attention/projection 的计算量占主导，Qwen3 的结构劣势显现。

## 6. 为什么 32 token Qwen3 更快、128 后反而更慢

本机数据更接近以下模型：

```text
latency = fixed runtime overhead
        + tokenization/prompt overhead
        + token-linear decoder work
        + sequence-dependent attention work
```

在 32 token：

- GPU kernel launch 和 request overhead 占比高；
- Qwen3 的 28-layer implementation、INT4 kernel 和调用路径在这台 GPU 上固定开销较低；
- Qwen3 约 28.8 ms，GME 约 32.7 ms。

在 128 token 及以上：

- Qwen3 的 attention projection/head 配置和 2048 hidden input 开始主导；
- GME 的 1536 hidden、2 KV head 配置更省；
- 结果变为 Qwen3 约 1.27x（128）、1.15x（512）、1.27x（1024）。

128 到 512 的比例不是单调的，说明 GPU clock state、kernel selection 和 dynamic shape
也会造成几百分点波动；但 1024 的稳定交替结果足以证明差距不是单纯的 C++/Python
测量错误。

## 7. 最终结论

1. **Qwen3 1024 比 GME Python 慢很多的现象在严格交替复测中仍存在，但差距应从旧表的
   “约 351.6 vs 279.1”修正为本次对齐口径的“约 356.0 vs 280.5”，约 1.27x。**
2. **GME C++ 的大幅波动确实是 GPU boost/throttle 和独立进程测量方式造成的，不能作为
   Qwen3 固定性能基准。**
3. **Qwen3 与 GME 不是同一个 decoder contract**：Qwen3 hidden=2048、8 KV heads、
   独立 text embedding IR；GME hidden=1536、2 KV heads、token embedding 集成在 language IR。
4. **32 token Qwen3 更快主要是固定调度/实现开销优势；长序列 Qwen3 变慢主要是 attention
   projection/head/hidden-state 计算和额外 inputs_embeds 路径成本。**
5. 当前不应把 Qwen3 的 1024 latency 归因于一次异常 GPU 状态；应以交替多轮 median/p10/p90
   作为对比口径。
