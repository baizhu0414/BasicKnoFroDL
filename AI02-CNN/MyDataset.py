# 如何将现有数据套到新的模型中？
# 数据组织
# train
#   \cats
#     \cat1.png
#     \cat2.png
#     ...
#   \dogs
#     \dog1.png
#     \dog2.png
#     ...
# test
#   ...
# Dataset：实现__init__, __getitem__,__len__
# ImageFolder:文件夹格式为train,val,test这种名称；train/dogs,train/cats这种分类; train/dogs/dog1.png这种数据
from torch.utils.data import Dataset # 数据加载
from PIL import Image # 图像打开
import os # 路径处理

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir)) # cats,dogs

        self.img_labels = []
        
        for idx,class_name in enumerate(self.classes): # idx表示分类：0，1
            class_dir = os.path.join(root_dir, class_name) # 分类路径
            # 遍历文件名称
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.img_labels.append((idx, img_path))

    def __getitem__(self, index): # 真正打开每一张图像
        idx, img_path = self.img_labels[index]
        # 使用PIL打开图片，并转换为RGB格式，避免灰度图或其他通道数
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # 返回处理后的图片和标签
        return img, idx
    
    def __len__(self):
        return len(self.img_labels)