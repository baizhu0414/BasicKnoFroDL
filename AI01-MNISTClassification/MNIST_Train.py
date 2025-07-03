import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

# 定义神经网络
class QYNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层:作用是整合特征
        self.fc1 = nn.Linear(28*28, 256) # 输入是灰度图，输出是256维神经元
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)


    def forward(self, x): # x是图片参数
        # 展平图像作为全连接输入(64,1,28,28)->(64,1*28*28)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) # 输出层不用relu。
        # x = torch.softmax(x) # 进行概念的映射，此处使用CELoss内部包含。
        return x # 每种分类的概率 

if __name__ == '__main__':
    torch.manual_seed(21) # 设置随机数种子，方便复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Train:使用{device}设备")

    transform = transforms.Compose([ # 仅用于图像预处理
        # 将图像转为张量并归一化到[0.0,1.0]
        transforms.ToTensor(),
        # 进一步标准化数据，提高训练效率x~(0,1);(x-0.5)/0.5~(-1,1)
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = datasets.MNIST("./data", 
                                train = True, # 下载训练集
                                transform=transform, 
                                download = True);
    test_dataset = datasets.MNIST("./data",
                                train=False,
                                transform=transform,
                                download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = QYNet().to(device)
    loss_fun = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器，用于梯度下降算法


    train_losses = []
    train_accs = []
    test_accs = []

    # 开始训练模型
    epochs = 10 # 轮次
    best_acc = 0.0
    best_model_path = "best_mnist_new_model.pth"

    for epoch in range(epochs):
        running_loss = 0.0 # 当前损失
        correct_train = 0 # 预测正确数
        total_train = 0 # 样本总数

        # 设置训练模式
        model.train()
        # labels:64张图片对应的类别组成数组
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad() # 梯度清零，防止累积
            outputs = model(imgs) # (64,10)每种分类的概率
            loss = loss_fun(outputs, labels) # 结果返回Tensor，损失计算本质上是按索引提取概率值后求平均，而非依赖广播。
            loss.backward() # 反向传播
            optimizer.step() # 参数更新
            # 提取标量值：将单元素张量转换为 Python 的 float 或 int。
            #释放计算图：断开与计算图的连接，释放显存并提高效率。
            running_loss += loss.item()

            properties, prediction = torch.max(outputs, 1) # (64,10)->(64,)
            total_train += len(labels) # 统计总样本图片数量
            correct_train += (prediction==labels).sum().item() # 预测准确的数量

        # 完成一轮训练，计算各种数据
        train_acc = correct_train / total_train
        train_accs.append(train_acc)
        # 计算当前轮次的平均损失
        # len(train_loader)表示"批次数"，每个批次计算一次loss求和，此处再平均。
        train_losses.append(running_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[epoch]}, Train Accuracy: {train_acc}")


        # 当前epoch model进行验证
        model.eval() # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, prediction = torch.max(outputs, 1) # 针对(64,10)第二维取最大
                total += len(labels)
                correct += (prediction == labels).sum().item()
            
        test_accs.append(correct / total)
        print(f"Epoch{epoch+1}/{epochs}, Test Accuracy:{test_accs[epoch]:.2%}")

        if test_accs[epoch] > best_acc:
            best_acc = test_accs[epoch]
            torch.save(model.state_dict(), best_model_path) # 保存权重
            print(f"Best model saved with accuracy:{best_acc}")

    print(f"Best accuracy on test set:{best_acc}")


    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))


    # 绘制损失曲线
    plt.subplot(1, 2, 1)  # 选择第一个子图，1 行 2 列布局，当前操作子图索引为 1 
    plt.plot(train_losses, label='Training Loss')  # 传入训练损失数据 train_losses，设置曲线标签为 Training Loss
    plt.xlabel('Epoch')  # 设置 x 轴标签为 Epoch（迭代轮次）
    plt.ylabel('loss')  # 设置 y 轴标签为 loss（损失值）
    plt.title('Training Loss over Epochs')  # 设置子图标题为 Training Loss over Epochs（迭代轮次上的训练损失）
    plt.legend()  # 添加图例，展示曲线标签
    plt.grid(True)  # 添加网格，让图像刻度线更清晰，便于观察数据趋势


    # 绘制训练集和测试集准确率曲线
    plt.subplot(1, 2, 2)  # 选择第二个子图，1 行 2 列布局，当前操作子图索引为 2 
    plt.plot(train_accs, label='Train Accuracy')  # 传入训练集准确率数据 train_accuracies，设置曲线标签为 Train Accuracy
    plt.plot(test_accs, label='Test Accuracy')  # 传入测试集准确率数据 test_accuracies，设置曲线标签为 Test Accuracy
    plt.xlabel('Epoch')  # 设置 x 轴标签为 Epoch（迭代轮次）
    plt.ylabel('Accuracy')  # 设置 y 轴标签为 Accuracy（准确率）
    plt.title('Train and Test Accuracy over Epochs')  # 设置子图标题为 Train and Test Accuracy over Epochs（迭代轮次上的训练与测试准确率）
    plt.legend()  # 添加图例，展示两条曲线的标签
    plt.grid(True)  # 添加网格


    # 保存图像
    plt.tight_layout()  # 自动优化子图布局，避免标签、标题等重叠
    plt.savefig('loss_and_accuracy_curves.png')  # 将绘制好的图像保存为 loss_and_accuracy_curves.png 文件
    plt.show()  # 弹出窗口展示绘制的图像

