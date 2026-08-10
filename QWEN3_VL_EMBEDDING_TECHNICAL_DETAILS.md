# Qwen3-VL-Embedding-2B 技术说明与数据流

本文详细说明当前 OpenVINO GenAI 2026.3 runtime 中
`Qwen3-VL-Embedding-2B` 的 `text`、`image` 和 `fused` 执行路径、输入输出 shape、
`M x N` 图片如何进入视觉塔，以及最终如何得到 native 2048 维 embedding。

> 模型名称说明：用户有时会简称为 `qwen2-vl-embedding-2b`，本项目实际模型是
> `Qwen3-VL-Embedding-2B`。它基于 Qwen3-VL 架构，OpenVINO 模型目录使用
> `Qwen3-VL-Embedding-2B-FP32/INT8/INT4`。

## 1. 模型和 OpenVINO 文件布局

官方导出任务：

```bash
optimum-cli export openvino \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --task feature-extraction \
  Qwen3-VL-Embedding-2B
```

当前转换目录包含：

| 文件 | 作用 | 典型输出 |
|---|---|---|
| `openvino_tokenizer.xml` / tokenizer files | 文本 tokenizer | `input_ids`, `attention_mask` |
| `openvino_text_embeddings_model.xml` | token embedding lookup | `[B,S,2048]` |
| `openvino_vision_embeddings_model.xml` | vision patch encoder | patch/vision hidden states |
| `openvino_vision_embeddings_pos_model.xml` | 2-D/3-D visual position interpolation | interpolated position features |
| `openvino_vision_embeddings_merger_model.xml` | visual attention、merger 和 deepstack features | merged visual embeddings |
| `openvino_language_model.xml` | Qwen3-VL language decoder + pooling postprocess | `[B,2048]` |
| `processor_config.json` / `preprocessor_config.json` | image/video processor 参数 | patch/grid 配置 |
| `chat_template.jinja` / tokenizer files | prompt/template 相关信息 | token IDs |

INT8/INT4 只改变权重精度；输入输出 contract 不变。当前 runtime 使用
`EmbeddingPipeline`，不是生成式 `VLMPipeline`。

## 2. 总体执行图

三种输入最终都汇聚到 language embedding pipeline：

```text
text/image/fused public API
        |
        v
format prompt + tokenize + normalize image inputs
        |
        +--> text embedding IR ------------------+
        |       input_ids [B,S]                  |
        |       output text_embeds [B,S,2048]    |
        |                                          v
        +--> optional vision IR --> visual tokens + deepstack features
                                                   |
                                                   v
                                        merge visual tokens into image_pad slots
                                                   |
                                                   v
                               language IR inputs_embeds [B,S,2048]
                               attention_mask [B,S]
                               position_ids [3,B,S]
                               visual_pos_masks [B,S]
                               deepstack_visual_embeds [3,visual_seq,2048]
                                                   |
                                                   v
                                Qwen3-VL decoder, 28 layers
                                                   |
                                                   v
                                last-token pooling -> [B,2048]
                                                   |
                                                   v
                                L2 normalize -> embedding [B,2048]
```

注意：不同 precision 的 IR 可能内部节点和权重表示不同，但 public output 都是
float32 native 2048 维。

## 3. Text-only 路径

### 3.1 Python API

当前 Python benchmark 调用：

```python
pipeline = openvino_genai.EmbeddingPipeline(model_dir, "GPU")
result = pipeline.embed("some text")
embedding = result.embeddings  # [1, 2048]
```

batch text 调用：

```python
result = pipeline.embed([text_a, text_b, text_c])
# result.embeddings.shape == [3, 2048]
```

### 3.2 Prompt 和 tokenization

`EmbeddingPipeline` 的 text-only 路径会：

1. 接收字符串或字符串列表；
2. 根据可选 `embedding_prompt` 组织 system/user prompt；
3. 对每条输入构造 token IDs；
4. 使用 right padding 组成 `[B,S]` 的 `input_ids` 与 `attention_mask`；
5. 通过 `openvino_text_embeddings_model.xml` 得到 `[B,S,2048]` token embeddings；
6. 构造纯文本所需的 Qwen3-VL 额外输入；
7. 运行 `openvino_language_model.xml`；
8. 用 attention mask 选择最后一个有效 token；
9. L2 normalize，返回 `[B,2048]`。

对 text-only，视觉数据不会运行视觉塔，也不会生成真实 image patches。为了满足
Qwen3-VL language IR 的输入 contract，patched runtime 会创建全 false 的：

```text
visual_pos_masks        [B,S]
deepstack_visual_embeds [3,S,2048]  # 当前模型的 deepstack layers = 3
```

原始 2026.3 代码错误地创建了 `[B,1]` 和 `[3,1,2048]`，导致 text-only
`Reshape_2` failure；修复详情见 [HISTORICAL_DIAGNOSIS.md](HISTORICAL_DIAGNOSIS.md)。

### 3.3 Text pooling

`openvino_language_model.xml` 的原始 decoder 输出在后处理前是：

```text
last_hidden_state [B,S,2048]
attention_mask    [B,S]
```

默认 `LAST_TOKEN` pooling 的等价逻辑是：

```python
last_index = attention_mask.sum(axis=1) - 1
pooled = last_hidden_state[range(B), last_index]  # [B,2048]
embedding = l2_normalize(pooled, axis=1)
```

OpenVINO GenAI 在 `text_embedding_utils.cpp` 中将 pooling 和 normalize 作为 output
postprocessing 加到模型上。它不是额外的 Python 循环，也不是生成 logits 后再恢复
hidden state。

## 4. Image-only 路径

### 4.1 Public input

Python image-only 调用：

```python
image = ov.Tensor(np.asarray(pixels_rgb_uint8))
result = pipeline.embed(images=[image])
# result.embeddings.shape == [1, 2048]
```

当前 benchmark 图片 tensor 约定为：

```text
shape = [M, N, 3]
dtype = uint8
layout = HWC
color = RGB
```

`M` 是原始高度，`N` 是原始宽度。它们不需要预先是 32 的倍数，也不需要调用方
手动 patchify；Qwen3-VL processor/runtime 会负责 resize、normalize 和 patch layout。

### 4.2 M x N 的 smart resize

Qwen3-VL 的视觉配置关键参数：

```text
patch_size          = 16
merge_size          = 2
temporal_patch_size = 2
min_pixels          = 4096
max_pixels          = 1310720  # 当前导出 processor_config 的 longest_edge
image mean          = [0.5, 0.5, 0.5]
image std           = [0.5, 0.5, 0.5]
rescale             = 1/255
```

runtime 的 `qwen2_vl_utils::smart_resize()` 对输入高度 `M`、宽度 `N` 和 factor=32
执行：

```text
H0 = round(M / 32) * 32
W0 = round(N / 32) * 32
```

然后约束像素面积：

```text
if H0 * W0 > max_pixels:
    beta = sqrt(M*N / max_pixels)
    H' = floor(M / beta / 32) * 32
    W' = floor(N / beta / 32) * 32

elif H0 * W0 < min_pixels:
    beta = sqrt(min_pixels / (M*N))
    H' = ceil(M * beta / 32) * 32
    W' = ceil(N * beta / 32) * 32

else:
    H' = H0
    W' = W0
```

其中 `H'`、`W'` 是送入 patch/grid 计算的 resized 尺寸。绝对宽高比超过 200:1
会被拒绝。`32 = patch_size * merge_size`，保证 spatial merge 后尺寸整除。

### 4.3 RGB 到 patch tensor

输入 `[M,N,3] uint8 RGB` 经过：

```text
RGB uint8 HWC
  -> resize to [H',W']
  -> float32
  -> rescale x (1/255)
  -> (pixel - [0.5,0.5,0.5]) / [0.5,0.5,0.5]
  -> temporal tile of 2 frames for an image
  -> patchify with 16x16 spatial patch
  -> spatial merge 2x2
```

单张 image 的 temporal dimension 为 1 个逻辑 frame，但 vision encoder 的
`temporal_patch_size=2` 会把 image branch 的输入复制/tile 成两个 temporal slices，
以满足 Qwen3-VL vision IR 的输入 contract。它不是把一张图片当作两张独立图片输出，
而是模型要求的 temporal patch packing。

raw patch 的二维 grid 为：

```text
grid_t = 1
grid_h = H' / 16
grid_w = W' / 16
```

raw spatial patch 数：

```text
P = grid_t * grid_h * grid_w = H'W'/256
```

经过 2x2 spatial merge 后，视觉 token 数：

```text
K = grid_t * (H'/32) * (W'/32)
  = H'W'/1024
```

有些 runtime 内部将 temporal tiled raw rows 计入 `pixel_values`；最终送入 language
prompt 的 image-pad 数量仍由 merge 后的视觉 token 数决定。Qwen3-VL 的导出/processor
metadata 会记录实际 grid；benchmark 结果以 runtime 真实 output 为准。

### 4.4 视觉 IR 流程

对一张图片，`InputsEmbedderQwen3VL` 的实际逻辑是：

```text
ov.Tensor [M,N,3] uint8
  -> VisionEncoderQwen2VL/Qwen3VL::encode
  -> openvino_vision_embeddings_model.xml
  -> vision patch hidden states
  -> openvino_vision_embeddings_pos.xml
  -> interpolated 2-D positional embeddings
  -> spatial merge + positional addition
  -> openvino_vision_embeddings_merger_model.xml
  -> visual embeddings + deepstack features
```

视觉 merger 输出两类数据：

```text
merged image embeddings       [K,2048]
deepstack feature lists        [3,K,2048]  # 3 selected visual layers
```

具体的 `K` 随每张图片的实际 resized grid 改变。不同图片可以有不同 `K`，因此 runtime
会保存每张图片的 grid 和 visual sequence，不能把所有图片简单 reshape 成同一固定
分辨率。

### 4.5 image token 注入

image-only 仍然需要一个文本 prompt，因为 Qwen3-VL language decoder 的输入是统一的
text+vision sequence。runtime 会在 prompt 中放入 image tag，并展开为：

```text
<vision_start>
<image_pad> <image_pad> ... <image_pad>   K times
<vision_end>
```

然后：

1. text embedding IR 先生成整个 prompt 的 text hidden states `[1,S,2048]`；
2. `input_ids` 中 image-pad 对应位置形成 `visual_pos_masks [1,S]`；
3. `merge_text_and_video_image_embeddings()` 将 `[K,2048]` visual embeddings 替换
   text hidden 中的 image-pad slots；
4. `deepstack_visual_embeds` 保留给 decoder 前几层的 DeepStack 注入；
5. language decoder 对统一序列做 attention；
6. last-token pooling + L2 normalization 返回 `[1,2048]`。

image-only 的最终 vector 不是单独对视觉 token 做平均；它是视觉 token 注入 Qwen3-VL
language sequence 后，由 decoder 最后有效 token hidden state 产生的跨模态 embedding。

## 5. Fused text + image 路径

### 5.1 输入

```python
result = pipeline.embed(
    "A text description or retrieval query",
    images=[image_tensor],
)
# [1,2048]
```

fused 的图像预处理与 image-only 完全相同，区别是 unified prompt 同时含有普通文本
和 image-pad token：

```text
system instruction
user:
  <vision_start> <image_pad> ... <image_pad> <vision_end>
  text content
```

实际 token 顺序由 Qwen3-VL chat template 和 `EmbeddingPipeline::format_prompt()`
决定；不要按字符串顺序假设 text 一定在 image 前，最终应以 tokenizer 的 `input_ids`
为准。

### 5.2 数据流

```text
text string + [M,N,3] image
        |
        +--> tokenize unified multimodal prompt -> input_ids [1,S]
        |
        +--> image smart_resize -> grid [1,H'/16,W'/16]
        |       -> visual tokens [K,2048]
        |       -> deepstack [3,K,2048]
        |
        +--> text token embeddings [1,S,2048]
        |
        +--> replace image_pad positions with visual embeddings
        |
        +--> Qwen3-VL language decoder [1,S,2048]
        |
        +--> last valid token + L2 normalize
                |
                +--> [1,2048]
```

fused 的成本通常由两部分组成：

```text
text decoder cost(sequence length including K image tokens)
+ visual encoder/merger cost(image patch count K)
```

所以 fused 不等于单纯的 `text_ms + image_ms`；视觉 token 会改变 language sequence
length，attention 和 decoder cost 也会改变。

## 6. Batch 和不同 M x N 图片

### 6.1 Text batch

```python
pipeline.embed([text_a, text_b, text_c])
```

runtime 会为每个 item 生成自己的 token hidden、position IDs 和 extra inputs，然后
按 `max_sequence_length` right-pad：

```text
inputs_embeds       [B,max_S,2048]
attention_mask      [B,max_S]
position_ids        [3,B,max_S]
visual_pos_masks    [B,max_S]
output              [B,2048]
```

### 6.2 Multimodal batch 的限制

当前 public API 的 `images=[...]` 是一组 images，`texts=[...]` 是一组 texts，但没有
显式的 `(text_i,image_i)` pair list contract。当前 patched runtime 的 batch extra-input
修复会按 batch item 保存/拼接 `visual_pos_masks` 和 `deepstack_visual_embeds`，因此
可以正确处理同图/同文本和 ragged text batch；但 image/fused 服务若需要严格的“一行一页”
语义，应由应用层逐页调用，或明确设计 pair-aware batching，而不是把多张图片一次性
附加给每个文本。

### 6.3 为什么 M x N 不要求固定 shape

OpenVINO vision IR 的 raw image inputs 使用动态 patch/grid shape；调用方可以传不同
分辨率。每张图片进入 runtime 后拥有自己的：

```text
resized_source_size = [H',W']
grid_thw             = [1,H'/16,W'/16]
visual_token_count   = K
```

多个视觉输入在内部按各自 `grid_thw` 切分和重排。对线上服务来说，真正需要控制的
不是原始像素 `M*N`，而是 resize 后的像素面积和最终视觉 token K；K 越大，视觉 attention
和 language sequence 都可能越慢。

## 7. 当前模型输出和质量边界

- native output：`[B,2048]` float32；
- 默认经过 last-token pooling 和 L2 normalize；
- MRL 低维使用是 output 后缀/前缀截断，不会让主干 IR 只计算低维；
- INT8/INT4 只改变模型权重精度，不改变上述 shape contract；
- 生产前应对 text/image/fused 分别做 native 2048 cosine parity 和真实检索质量回归。

## 8. 与 GME 技术说明的对应关系

| GME 说明 | Qwen3-VL-Embedding 对应 |
|---|---|
| GME `gme_openvino.py` | GenAI `EmbeddingPipeline` + Qwen3 `InputsEmbedder` |
| GME language IR | Qwen3 `openvino_language_model.xml` |
| GME visual IR | Qwen3 vision embeddings + pos + merger 三个 IR |
| GME 1536 hidden/output | Qwen3 2048 hidden/output |
| GME image grid `H/14,W/14` | Qwen3 grid `H'/16,W'/16`，2x2 merge 后 K tokens |
| GME text/image/fused | Qwen3 text/image/fused，最终都进入 language decoder pooling |
| GME batch mode | Qwen3 patched batch extra-input aggregation |

## 9. 相关源码和数据

- [EmbeddingPipeline](../benchmark/openvino.genai/src/cpp/src/rag/embedding_pipeline.cpp)
- [Qwen3-VL inputs embedder](../benchmark/openvino.genai/src/cpp/src/visual_language/qwen3_vl/classes.cpp)
- [Qwen2/Qwen3 vision preprocessing](../benchmark/openvino.genai/src/cpp/src/visual_language/qwen2vl/classes.cpp)
- [processor_config.json](../benchmark/models/Qwen3-VL-Embedding-2B-INT8/processor_config.json)
- [preprocessor_config.json](../benchmark/models/Qwen3-VL-Embedding-2B-INT8/preprocessor_config.json)
- [benchmark results](benchmark_results/BENCHMARK_RESULTS.md)
