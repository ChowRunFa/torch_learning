import torch
from torch import nn

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

# inputs = torch.reshape(input,(1,1,1,3))
# targets = torch.reshape(target,(1,1,1,3))

loss = nn.L1Loss()
result = loss(input,target)

loss_mse = nn.MSELoss()
result_mse = loss_mse(input,target)


print(result)
print(result_mse)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])

x = torch.reshape(x,(1,3))
loss_cross_entropy = nn.CrossEntropyLoss()
result_cross = loss_cross_entropy(x,y)
print(result_cross)