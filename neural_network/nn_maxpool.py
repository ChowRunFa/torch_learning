import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]],dtype=torch.float)

input = torch.reshape(input,(-1,1,5,5))

print(input)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output



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