import torch.nn as nn

# 定义一个简单的卷积神经网络用于猫狗分类任务
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取模块：通过三层卷积 + 激活 + 池化提取图像的空间特征
        # 神经网络层（如 nn.Conv2d、nn.MaxPool2d 等）默认支持批量处理batch
        self.features = nn.Sequential(
            # 第1个卷积层：输入3通道（RGB），输出16通道，卷积核大小为3x3，padding=1保持输出尺寸
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数，引入非线性
            nn.MaxPool2d(2),  # 缩小一半尺寸（224 → 112）
            # 第2个卷积层：输入16通道，输出32通道
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化后尺寸 112 → 56
            # 第3个卷积层：输入32通道，输出64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化后尺寸 56 → 28 （C*H*W=64*28*28）
        )

        # 分类器模块：全连接层将提取的特征进行分类
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),  # 输入为展平后的特征张量，输出128维
            nn.ReLU(),                      # 非线性激活
            nn.Dropout(0.5),                # Dropout防止过拟合，随机丢弃50%神经元
            nn.Linear(128, 2)               # 输出层：2个神经元表示猫（0）或狗（1）
        )

    def forward(self, x):
        x = self.features(x)                # 先通过卷积提取特征
        x = x.view(x.size(0), -1)           # 将多维特征图展平为1维向量（batch_size, C*H*W）
        x = self.classifier(x)              # 再送入全连接层进行分类,结果是Tensor
        return x