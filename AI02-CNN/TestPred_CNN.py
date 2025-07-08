import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from cnnModel import SimpleCNN  # 从model.py 导入自己定义的CNN模型类

# 选择设备：优先使用GPU，如果没有就使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型结构和权重
model = SimpleCNN().to(device)  # 将模型放到对应设备
model.load_state_dict(torch.load(r"AI02-CNN\cat_dog_cnn_new.pth", map_location=device))  # 加载模型参数
model.eval()  # 切换为推理模式，关闭Dropout、BatchNorm中的训练行为

# 类别名称：0表示猫，1表示狗
classes = ['cat', 'dog']

# 定义图像预处理流程：包括缩放、裁剪、归一化等
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 先将图像缩放到256x256
    transforms.CenterCrop(224),  # 然后从中间裁剪出224x224区域
    transforms.ToTensor(),  # 转换为Tensor，并将像素归一化到[0, 1]
    transforms.Normalize([0.5, 0.5, 0.5],  # 再对RGB三个通道做标准化，中心为0
                         [0.5, 0.5, 0.5]) # mean,std分别为0.5
])


# 指定要进行推理的图像路径（请根据实际修改）
img_path = r"AI02-CNN/data/test/cats/cat.4002.jpg"  # 示例路径，可替换为其他图片
img = Image.open(img_path).convert('RGB')  # 打开图像，并确保为RGB三通道

# 保存一份原始图像，用于后续展示（不影响推理）
display_img = img

# 对图像做预处理，添加 batch 维度后送入模型
img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224):适配model输入
# 自动判断真实标签（根据文件名中的 "cat" 或 "dog" 关键字）
if 'cat' in img_path.lower():
    true_class = 0
elif 'dog' in img_path.lower():
    true_class = 1
else:
    true_class = -1  # 如果文件名中不包含这两个词，设为未知

# 模型推理过程（关闭梯度计算，提高推理效率）
with torch.no_grad():
    outputs = model(img_tensor)  # 获取模型输出
    _, predicted = torch.max(outputs, 1)  # 取最大概率对应的类别索引
    pred_class = predicted.item()  # 从tensor中取出int类型结果

# 打印预测结果和真实标签（如果能判断出来）
print(f"真实标签: {classes[true_class] if true_class != -1 else '未知'}")
print(f"预测结果: {classes[pred_class]}")

# 构造图像标题内容（包括预测和真实值）
title = f"Predicted: {classes[pred_class]}"
if true_class != -1:
    title += f" | Ground Truth: {classes[true_class]}"
    # 正确预测用绿色标题，否则红色
    color = 'green' if pred_class == true_class else 'red'
    # 这里可补充基于 color 对标题样式渲染的代码，比如用 matplotlib 展示图像时设置标题颜色
    # 示例（需结合实际可视化库，如matplotlib）：
    # import matplotlib.pyplot as plt
    # plt.title(title, color=color)
else:
    color = 'black'  # 如果无法判断真实标签，标题为黑色

# 使用matplotlib绘图展示
plt.imshow(display_img)       # 显示图像
plt.title(title, color=color) # 设置图像标题，颜色由 color 变量控制
plt.axis('off')               # 去掉坐标轴
plt.show()                    # 展示图像窗口