import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    """
        如果输入 t 不是元组（比如是整数、字符串等其他类型），则返回一个由 t 重复两次构成的元组 (t, t)。
    """
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    """
        参考 cats_and_dogs.ipynb 中的 ViT 引用进行解释。

        image_size=224：输入图像的高度和宽度。（允许int, (int,int)）
        patch_size=32：每个 patch 的大小，用于将图像分割为固定大小的块。
        num_classes=2：分类任务的类别数（二分类）。
        dim=128：Transformer 中 token 的特征维度，与 Linformer 的维度匹配。
        transformer=efficient_transformer：自定义的 Linformer 编码器，用于处理序列。
        pool='cls'：池化方式，选择使用 CLS token 进行分类。
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        image_size_h, image_size_w = pair(image_size)
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # 49 patches for 224x224 image with 32x32 patch size
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        # 32x32 patch with 3 channels, so patch_dim = 3 * 32 * 32 = 3072
        patch_dim = channels * patch_size ** 2

        # 将图像转换为 patch embeddings。
        self.to_patch_embedding = nn.Sequential(
            # 将输入图像 b c h w - [B, 3, 224, 224] 重排为 [B, 49, 3072]，即 49 个 patches，每个 patch 维度 c*p1*p2=3072。
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim), # 	对每个 patch 的 3072 维特征进行归一化。
            nn.Linear(patch_dim, dim), # 通过线性投影 (3072, 128) 将 patch 特征降维到 128 维，输出 [B, 49, 128]。
            nn.LayerNorm(dim) # 再次归一化，稳定后续训练。
        )

        # 可学习的位置编码矩阵：(1,50,128) - 看成 50 行，每行 128 列。
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 额外的 cls token 用于分类任务(1,1,128) - 看成一行
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 使用传入的 efficient_transformer（即 Linformer）处理序列。
        self.transformer = transformer

        self.pool = pool # 'cls'
        self.to_latent = nn.Identity() # 恒等映射，保留扩展性

        self.mlp_head = nn.Sequential( # 参数（b,128）
            nn.LayerNorm(dim), # 归一化
            nn.Linear(dim, num_classes) # 全连接
        )

    def forward(self, img):
        # （b,49,128）
        x = self.to_patch_embedding(img)
        # b表示batch size，n表示patch数量，_表示特征维度
        b, n, _ = x.shape

        # 在每个 batch 的开头添加 cls token，变成 (b, 1, d)
        # (1,1,128) -> (b,1,128)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1) # 追加cls，变成 (b, 49 + 1, 128)
        x += self.pos_embedding[:, :(n + 1)] # 加上位置编码(1,50,128) -> (b, 50, 128)
        x = self.transformer(x) # MLP Head：(b, 50, 128)

        # cls：(b,128)直接取序列的第一个 token（通常是 [CLS] token），输出形状为 (b, 128)。
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x) # 输入输出不变，可插入Linear层或其他操作(b, 128)
        return self.mlp_head(x) # 前馈神经网络(b, num_classes)
