import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 方式1  模型结构+模型参数
torch.save(vgg16, 'vgg16_method1.pth')

# 方式2  模型参数（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = Net()
torch.save(net,'network_method1.pth')