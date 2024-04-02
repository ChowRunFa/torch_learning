import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader




class Net_Linear(nn.Module):
    def __init__(self):
        super(Net_Linear,self).__init__()
        self.linear1 = nn.Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

if __name__ == '__main__':

    dataset = torchvision.datasets.CIFAR10(root='../torchvision/dataset', train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    net = Net_Linear()

    for data in dataloader:
        images, labels = data
        print(images.shape)
        # output =  torch.reshape(images,(1,1,1,-1))
        output = torch.flatten(images)
        print(output.shape)
        output = net(output)
        print(output.shape)