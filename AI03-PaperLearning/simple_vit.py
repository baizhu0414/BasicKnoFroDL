import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    # 生成二维位置编码
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    
    # 显式设置数据类型
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    # 前馈神经网络
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # 实现了多头自注意力机制算法
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    # 实现了 Transformer 的编码器部分，由多个注意力层和前馈网络层堆叠而成，并且每层都采用了残差连接。
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        # dim_head 仅用于内部计算，最终会被合并回原始的 dim 维度
        # 参数计算：dim = dim_head × heads
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                # mlp_dim仅用于内部计算
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            """
            注意力计算：每个 patch 都与其他所有 patch 计算注意力权重，捕获全局依赖关系。
            残差连接：将注意力输出与输入相加，防止梯度消失并保留原始信息。
            """
            x = attn(x) + x # 自注意力残差连接
            x = ff(x) + x # 前馈网络残差连接
        return self.norm(x)

class SimpleViT(nn.Module):
    """
        图像分块与嵌入：把输入图像分割成多个不重叠的 patch，接着将每个 patch 展平并通过线性层映射到指定维度。
        添加位置编码：为每个 patch 嵌入添加位置编码，使模型能够感知 patch 之间的相对位置。
        Transformer 处理：利用 Transformer 编码器对 patch 序列进行处理。
        分类头：对所有 patch 的特征进行平均池化，然后通过线性层得到分类结果。
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            # 分割展平图像：[B, num_patches, patch_dim] 
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            # 映射到指定维度：[B, num_patches, dim]
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height, # patch 数量
            w = image_width // patch_width, # patch 数量
            dim = dim, #(h×w, dim)=(num_patches, dim)
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img) # (B, num_patches, dim)
        # pos_embedding: (num_patches, dim)-> 广播，输出 (B, num_patches, dim)
        x += self.pos_embedding.to(device, dtype=x.dtype) 

        x = self.transformer(x) # 输出 x 是融合了全局上下文信息的增强特征 (B, num_patches, dim)
        x = x.mean(dim = 1) # 沿着维度 1（即 patch 序列维度）进行全局平均池化，(B, dim)

        x = self.to_latent(x)
        return self.linear_head(x) # 全连接得到分类结果 (B, num_classes)
