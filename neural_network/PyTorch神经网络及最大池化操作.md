
# 使用 PyTorch 的卷积神经网络和 CIFAR-10 数据集

这段脚本展示了如何创建一个包含单个最大池化层的简单卷积神经网络（CNN）。该网络将被用于 CIFAR-10 数据集。代码还包括了与 TensorBoard 的集成，以便可视化最大池化层的输入和输出。

## 导入模块

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
```

这些是脚本中使用的主要 PyTorch 和 torchvision 库，以及用于数据加载和TensorBoard可视化的模块。

## 输入数据

```python
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float)
input = torch.reshape(input, (-1, 1, 5, 5))
print(input)
```

这里定义了一个5x5的二维张量，并将其转换为一个4维张量，这是为了模拟单通道图像的批次。`-1`在`reshape`函数中表示自动计算该维度的大小，这里表示批次大小为1。

## 网络定义

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output
```

定义了一个继承自`nn.Module`的网络类`Net`。在构造函数中，定义了一个最大池化层，其核大小为3。在前向传播函数`forward`中，输入数据通过最大池化层进行处理。

## 主函数

```python
if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../torchvision/dataset', train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    writer = SummaryWriter('logs')

    net = Net()

    output = net(input)
    print(output)
    step = 0
    for data in dataloader:
        img, label = data
        writer.add_images('input',img,step)
        output = net(img)
        writer.add_images('output',output,step)
        print('step{}:'.format(step))
        step += 1
    writer.close()
```

在主函数中，首先加载 CIFAR-10 数据集，并利用 DataLoader 进行批处理。

然后，初始化 TensorBoard 的 SummaryWriter，以便记录和可视化数据。

接着，创建`Net`类的实例，并对定义好的输入数据`input`进行前向传播，打印输出结果。

最后，对 DataLoader 中的每个批次数据执行以下步骤：

1. 将输入图像写入 TensorBoard。
2. 将输入图像通过网络进行前向传播，获取输出。
3. 将输出图像写入 TensorBoard。
4. 打印当前步骤。

这个循环将帮助我们理解网络在处理不同批次的图像时的行为。

## TensorBoard 可视化

在脚本执行后，可以使用以下命令启动 TensorBoard：

```bash
tensorboard --logdir=logs
```

这将允许我们在浏览器中查看输入和输出的图像，以及网络在训练过程中的其他可能的指标。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/3f9a5f8479dd420b8521710d31a60b15.png)

