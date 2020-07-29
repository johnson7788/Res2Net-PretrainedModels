from torchvision import models
import torch
from PIL import Image
#图片预处理包
from torchvision import transforms

######################################################
# 如何使用torchvision的alexnet模型
######################################################

alexnet = models.alexnet(pretrained=True)
print(alexnet)
#预处理,使得图片具有正确的形状和其他特征，例如均值和标准差。这些值应与训练模型时使用的值相似。这样可以确保网络会产生有意义的答案
transform = transforms.Compose([               #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
    )])
"""
第[1]行：这里我们定义了一个变量变换，它是对输入图像进行的所有图像变换的组合。
第[2]行：将图像尺寸调整为256×256像素。
第[3]行：将图像裁剪为围绕中心的224×224像素。
第[4]行：将图像转换为PyTorch Tensor数据类型。
[5-7行]：通过将图像的均值和标准差设置为指定值来对图像进行归一化。
"""
#使用torchvision支持的pillow模块加载图片
# Import Pillow, 拉布拉多犬
img = Image.open("dog.jpg")

#进行预处理 img_t  [3,224,224]
img_t = transform(img)
#加一个维度作为批次, batch_t [1,3,224,224]
batch_t = torch.unsqueeze(img_t, 0)

#模型评估
alexnet.eval()
#获取输出预测结果
out = alexnet(batch_t)
print(out.shape)

#输出类别的名字
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

#拉布拉多犬
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
# 208, Labrador_retriever 41.58518600463867

#top 5, 前5的概率
_, indices = torch.sort(out, descending=True)
res = [(labels[idx], percentage[idx].item()) for idx in indices[0]]
print(res)
