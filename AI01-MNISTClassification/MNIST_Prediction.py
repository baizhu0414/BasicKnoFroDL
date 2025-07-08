from torchvision import transforms,datasets
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MNIST_Train import QYNet # 在 Python 中，导入模块时会执行被导入文件的全局代码

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)) # 标准化，mean,std分别为0.5
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # 下载测试集
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)  # 数据加载器：每批次只加载10张图片

print(f"总数据集{len(test_dataset)}, batch=10的批次数量{len(test_loader)}")

# 开始推理
model = QYNet()
best_model_path = "best_mnist_new_model.pth"

# 模型加载
model.load_state_dict(torch.load(best_model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


examples = enumerate(test_loader) # 带索引迭代器：(idx, (imgs, labels))，idx~(0,batch)

batch_idx, (imgs, labels) = next(examples)
imgs, labels = imgs.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(imgs)  
    _, predicted = torch.max(outputs, 1)  # 返回输入张量在指定维度上的最大值和对应索引。


fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(imgs[i].cpu().squeeze(), cmap='gray') # 去除多余维度squeeze:(1,28,28)->(28,28)
    axes[i].set_title(f"Pred: {predicted[i].item()}")  
    axes[i].axis('off')  
plt.show()


print("True labels:", labels.cpu().numpy()) # 一定要切换到CPU才能进行numpy处理
print("Predicted labels:", predicted.cpu().numpy())
