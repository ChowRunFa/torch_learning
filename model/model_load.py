import torch
import torchvision
from torch import nn

# 方式1 加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2  加载模型
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

from model_save import Net
model = torch.load('network_method1.pth')
print(model)