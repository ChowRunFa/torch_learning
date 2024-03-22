import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True,transform=dataset_transforms, download=False)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transforms,download=False)

# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
#
# print(test_set[0])

writer = SummaryWriter('logs')
for i in range(10):
    img,target = test_set[i]
    writer.add_image('test'+str(i),img,i)