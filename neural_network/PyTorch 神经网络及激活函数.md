

# PyTorch 神经网络及激活函数
这个脚本展示了如何使用 PyTorch 构建一个简单的神经网络，并在 CIFAR-10 数据集上进行训练。同时，该脚本展示了如何使用 TensorBoard 进行训练过程中输入和输出的可视化。

## 导入必要的库

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
```

这段代码导入了 PyTorch 核心库、视觉相关的 torchvision 库、神经网络模块、数据加载器 DataLoader，以及 TensorBoard 的 SummaryWriter 用于记录训练过程。

## 神经网络定义

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self,input):
        output = self.relu1(input)
        output = self.sigmoid1(input)
        return output
```

这里定义了一个名为 `NeuralNetwork` 的神经网络类，该类继承自 `nn.Module`。网络包含了一个 ReLU 激活层和一个 Sigmoid 激活层。在前向传播 `forward` 方法中，输入数据先后通过这两个激活层。但这里有一个错误，ReLU激活后的结果没有被使用，而是直接对原始输入使用了 Sigmoid 激活。

## 主程序

```python
if __name__ == '__main__':
    input = torch.tensor([[1, -0.5],
                          [-1, 3]])
    output = torch.reshape(input, (-1, 1, 2, 2))
    print(output.shape)
    net = NeuralNetwork()
    dataset = torchvision.datasets.CIFAR10(root='../torchvision/dataset', train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    writer = SummaryWriter('logs')
    step = 0
    for data in dataloader:
        images, labels = data
        writer.add_images('input', images, step)
        output = net(images)
        writer.add_images('output', output, step)
        step += 1
    writer.close()
```

在主程序中，首先定义了一个输入张量，并将其重新塑形为适合神经网络处理的形状。

然后实例化定义的 `NeuralNetwork` 类。

接下来，使用 `torchvision.datasets.CIFAR10` 加载 CIFAR-10 数据集，并通过 `DataLoader` 进行批处理。

初始化了 `SummaryWriter`，它将用于将数据写入 TensorBoard。

在一个循环中，遍历 `DataLoader` 获取数据批次，并使用 `writer.add_images` 将输入和输出的图像写入 TensorBoard。

> `add_images` 方法在写入输出图像时，需要确保图像数据在 `[0, 1]` 范围内，否则可能无法正确显示，而 Sigmoid激活可以保证这一点。

最后，关闭 `SummaryWriter`。

## TensorBoard 可视化

运行脚本后，使用以下命令启动 TensorBoard：

```bash
tensorboard --logdir=logs
```

这将在浏览器中启动 TensorBoard，可以查看记录的图像和其他指标。



![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4d060a0e936d48849564c8e91834f4c8.png)
