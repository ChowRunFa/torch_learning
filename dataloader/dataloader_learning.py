import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10(root='../torchvision/dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

img, target = test_dataset[0]
print(img.shape)
print(target)

writer = SummaryWriter('logs')
step = 0

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('Epoch{}:'.format(epoch),imgs,step )
        step += 1

writer.close()