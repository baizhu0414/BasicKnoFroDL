from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from cnnModel import SimpleCNN
# from torchvision.datasets import ImageFolder
from MyDataset import CatDogDataset
import os

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

# 创建训练集对象，加载路径data2\train下的图像，应用训练时的数据增强和预处理（如随机裁剪、翻转等）
train_dataset = CatDogDataset(r"AI02-CNN\data\train", transform=train_transform)
# 创建验证集对象，加载路径data2\test下的图像，应用验证时的固定预处理（如中心裁剪），保证验证结果稳定
test_dataset = CatDogDataset(r"AI02-CNN\data\test", transform=train_transform)

# 创建数据加载器
# DataLoader就是用来把你的数据集“打包好，一批一批地送进模型里”的工具
train_loader = DataLoader(
    train_dataset,       # 传入训练数据集对象
    batch_size=32,       # 每个批次加载32张图像
    shuffle=True,        # 打乱数据顺序，有利于模型更好地泛化
    num_workers=0        # 用于数据加载的子进程数量，0表示在主进程加载数据
)

test_loader = DataLoader(
    test_dataset,         # 传入验证数据集对象
    batch_size=32,       # 验证时也用相同批大小
    shuffle=False,       # 验证时不打乱数据，保证结果稳定可复现
    num_workers=0        # 同样在主进程加载数据
)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型、损失函数、优化器
model = SimpleCNN().to(device)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, test_loader, loss_fun, optimizer, epochs=10):
    # 训练多个轮次（epoch）
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式，启用dropout等
        train_loss, correct, total = 0, 0, 0  # 初始化训练损失、正确预测数量、总样本数
        # 遍历训练集的所有批次
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # 将当前batch的数据移动到GPU（如果使用GPU训练）
            images, labels = images.to(device), labels.to(device)
            # 前向传播：将图像输入模型，得到预测结果
            outputs = model(images)
            # 计算损失
            loss = loss_fun(outputs, labels)
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播：计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 累加损失和正确预测数量
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1) # 对第二维选最大值：(64,10)->(64,)
            total += images.size(0)
            correct += (predicted == labels).sum().item()

        # 计算平均损失和训练集准确率
        avg_loss = train_loss / total
        train_acc = correct / total

        # -------------验证阶段--------------
        # 在评估阶段不需要计算梯度，加速并节省显存
        with torch.no_grad():
            test_loss = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # 前向传播得到预测
                outputs = model(images)
                loss = loss_fun(outputs, labels)
                # 累加验证损失
                test_loss += loss.item() * images.size(0) # int*number
                # 获取预测类别
                _, predicted = torch.max(outputs, 1)
                # 累加验证集中总数与正确数
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算验证准确率
        test_acc = 100 * correct / total
        print(f"Test Loss: {test_loss/total:.4f}, Accuracy: {test_acc:.2f}%\n")

# 调用训练函数：传入模型、训练/验证数据加载器、损失函数、优化器和训练轮次
train_model(model, train_loader, test_loader, loss_fun, optimizer, epochs=10)
# 将训练好的模型参数保存到本地文件，方便之后加载使用或部署
torch.save(model.state_dict(), r"AI02-CNN\cat_dog_cnn_new.pth")
