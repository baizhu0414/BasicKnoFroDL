from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import SimpleCNN
from torchvision.datasets import ImageFolder
from MyDataset import CatDogDataset

# 定义训练集的图像预处理（transform）操作
# transforms.RandomRotation(90) 随机旋转图像，用于数据增强
# transforms.RandomHorizontalFlip(p=0.5) 以50%概率水平翻转图像，用于数据增强
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图片缩放到统一大小 256x256，方便批量训练
    transforms.RandomCrop(224),     # 随机裁剪一个224x224的区域，增强模型的鲁棒性
    transforms.ToTensor(),          # 将 PIL 图像或 ndarray 转换为 Tensor，并归一化到 [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],  # 对三个通道分别做标准化：x=(x-mean)/std，这样处理后像素值范围变为
                         [0.5, 0.5, 0.5])  # 让数据分布更均匀，减轻训练时梯度更新的震荡，有助于模型更快收敛
])