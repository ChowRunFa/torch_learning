

# PyTorch 神经网络及卷积操作详细笔记

本文档提供了使用 PyTorch 构建神经网络和执行卷积操作的详细步骤，包括如何加载数据集、定义模型、执行前向传播、以及利用 TensorBoard 进行结果的可视化。

## 导入必要的库

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
```

- `torch`: PyTorch 的核心库，提供了多维数组对象（张量）和自动求导功能。
- `torchvision`: 一个用于处理图像和视频的 PyTorch 库，提供了常见的数据集和模型。
- `nn`: 用于构建神经网络的模块，提供了构建深度学习模型所需的所有构件。
- `DataLoader`: 用于包装数据集并提供批量处理、打乱数据和多线程加载等功能。
- `SummaryWriter`: 用于将数据写入 TensorBoard，一个可视化工具，可以用来展示网络图、指标图和其他分析结果。

## 定义简单的神经网络

```python
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
```

在 `MyNeuralNetwork` 类中，我们定义了一个非常基础的神经网络结构，它继承自 `nn.Module`。`forward` 方法定义了数据如何通过网络流动。这里我们进行了一个简单的操作：将输入数据的每个元素增加1。

## 卷积操作的实现

```python
def nn_conv(self):
    # ...（省略了之前的代码）
    import torch.nn.functional as F

    # ...（省略了之前的代码）

    output = F.conv2d(input, kernel, stride=1)
    output = F.conv2d(input, kernel, stride=2)
    output = F.conv2d(input, kernel, stride=1, padding=1)
    print(output)
```

`nn_conv` 方法展示了如何在 PyTorch 中执行卷积操作。我们使用 `torch.nn.functional.conv2d` 函数进行卷积，该函数接受输入张量和卷积核张量，并通过 `stride` 和 `padding` 参数控制卷积的步长和填充。

- `stride`: 卷积核滑动的步长。
- `padding`: 在输入张量的边界上填充的值的数量。

## 加载 CIFAR10 数据集

```python
dataset = torchvision.datasets.CIFAR10(root='../torchvision/dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

这段代码使用 `torchvision` 提供的 `datasets` 模块来加载 CIFAR10 数据集，这是一个包含了10个类别的60000张32x32彩色图像的数据集。`transform=torchvision.transforms.ToTensor()` 确保数据被转换为 PyTorch 张量。`DataLoader` 将这个数据集包装成一个迭代器，便于批量训练。

## 定义一个卷积神经网络并进行训练

```python
class MyImgConvNet(nn.Module):
    # ...（省略了之前的代码）

if __name__ == '__main__':
    myimgconvnet = MyImgConvNet()
    print(myimgconvnet)

    writer = SummaryWriter('logs')
    step = 0
    for data in dataloader:
        img, label = data
        output = myimgconvnet(img)
        writer.add_image('input', img, step, dataformats="NCHW")
        output = torch.reshape(output, (-1, 3, 30, 30))
        writer.add_images('output', output, step, dataformats="NCHW")

        step += 1
```

在主函数中，我们实例化了 `MyImgConvNet` 类，它包含了一个卷积层。我们遍历 `DataLoader`，将每个批次的图像通过网络进行前向传播，并使用 `SummaryWriter` 将输入和输出的图像记录下来，以便在 TensorBoard 中进行可视化。`add_image` 和 `add_images` 方法允许我们记录标量、图像、音频和直方图等信息。

## TensorBoard 可视化

为了在 TensorBoard 中查看训练过程和结果，我们可以在终端中运行以下命令，然后在浏览器中打开提供的链接。

```bash
tensorboard --logdir=logs
```

TensorBoard 提供了一个交互式界面，可以查看模型结构、监控训练进度、分析模型参数等。这对于模型调试和理解是非常有用的。通过记录的图像，我们可以直观地看到模型是如何看待输入数据的，以及卷积层是如何处理数据的。