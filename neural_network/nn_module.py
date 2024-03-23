import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

    def sample_calcu(self):
        x = torch.tensor([1, 2, 3])
        output = self(x)
        print(output)

    def nn_conv(self):
        input = torch.tensor([[1, 2, 0, 3, 1],
                              [0, 1, 2, 3, 1],
                              [1, 2, 1, 0, 0],
                              [5, 2, 3, 1, 1],
                              [2, 1, 0, 1, 1]])
        kernel = torch.tensor([[1, 2, 1],
                               [0, 1, 0],
                               [2, 1, 0]])
        input = torch.reshape(input, [1, 1, 5, 5])
        kernel = torch.reshape(kernel, [1, 1, 3, 3])

        import torch.nn.functional as F


        print(input.shape)
        print(kernel.shape)

        output = F.conv2d(input,kernel,stride=1 )
        output = F.conv2d(input,kernel,stride=2 )
        output = F.conv2d(input,kernel,stride=1,padding=1 )
        print(output)

class MyImgConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x



if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../torchvision/dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    myimgconvnet = MyImgConvNet()
    print(myimgconvnet)

    writer = SummaryWriter('logs')
    step = 0
    for data in dataloader:
        img,label = data
        output = myimgconvnet(img)
        print(img.shape)
        print(output.shape)
        #torch.Size([64, 3, 32, 32])
        writer.add_image('input',img,step,dataformats="NCHW")
        #torch.Size([64, 6, 30, 30]) -> [xx, 3, 30, 30]
        output = torch.reshape(output,(-1,3,30,30))
        writer.add_images('output',output,step,dataformats="NCHW")

        step += 1