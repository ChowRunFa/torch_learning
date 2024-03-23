import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self,input):
        output = self.relu1(input)
        output = self.sigmoid1(input)
        return output



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


