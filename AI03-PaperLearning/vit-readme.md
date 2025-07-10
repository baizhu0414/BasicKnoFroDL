<img src="./images/vit.gif" width="500px"></img>

```python
efficient_transformer = Linformer(
    dim=128,
    # 公式：seq_len = (image_size // patch_size)² + 1
    seq_len=49+1,  # 序列长度（7×7 个图像 patch + 1 个分类 token）
    depth=12, # 编码器的深度（Transformer 的层数）
    heads=8,
    k=64 # 表示将 key/value 投影到的低维空间维度
)

model = ViT(
    dim=128,               # 1. 特征维度 与 Linformer 保持一致（必须匹配）
    image_size=224,        # 2. 输入图像尺寸
    patch_size=32,         # 3. 图像补丁大小：较大 patch，减少序列长度，适合轻量级任务
    num_classes=2,         # 4. 分类类别数：二分类任务
    transformer=efficient_transformer,  # 5. 自定义Transformer编码器
    channels=3,            # 6. 输入图像通道数
).to(device)              # 7. 加载到计算设备
```

按照上面的流程图，一个ViT block可以分为以下几个步骤

1. Patch embedding：
    - 输入图片大小为224 × 224，将图片分为固定大小的patch，patch大小为32 × 32，则每张图像会生成49个patch，即输入序列长度为49，每个patch维度32 × 32 × 3 = 3072.
    - “线性投射层” 的作用是对每个 Patch 的原始特征进行维度映射（类似 NLP 中词向量的嵌入过程）
    - 线性投射层的维度为3072 × N ( N = 128 )，因此输入通过线性投射层之后的维度为49 × 128【(49,3072)*(3072,128)】 ，即一共有 49个token，每个token的维度是128。
    - 在前面加上一个特殊字符cls，cls token 是一个额外的可学习向量（维度 128，随机生成），用于最终分类任务的 “聚合信息”。因此最终的维度是50 × 128。到目前为止，已经通过patch embedding将一个视觉问题转化为了一个seq2seq问题。【传统是2D张量的卷积处理，但是此处通过 patch embedding转变成了49个token，再加一个cls token。此时，视觉信息被编码为与 NLP 中 “文本序列”（如句子的词向量序列）结构一致的形式。】
2. Positional encoding(standard learnable 1D position embeddings)：
    - ViT需要加入位置编码，位置编码可以理解为一张表，表一共有N行，N的大小和输入序列长度相同，此处为50，每一行代表一个向量，向量的维度和输入序列embedding的维度相同(128)。
    - 注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是50 × 128。
3. LN/multi-head attention/LN：
    - LayerNorm（LN）：对输入序列（维度 50×128）进行层归一化
    - 输入维度为 50×128，首先通过线性层将其映射为查询（Q）、键（K）、值（V）。
    - 若注意力头数为 h（需满足 128 % h == 0，例如 h=8，则每个头的维度为 128/8=16），则每组 Q、K、V 的维度为 50×16，共 8 组。
    - 每个头独立计算自注意力后，将 8 组输出拼接，得到维度为 50×(16×8)=50×128 的结果。
4. MLP：
    - 前馈神经网络（MLP）通常包含 “升维→激活→降维” 的过程。
    - 将维度放大再缩小回去，50x128放大为50x512，再缩小变为50x128
    - 最终输出维度仍为 50×128，与输入 Transformer 的特征维度保持一致。