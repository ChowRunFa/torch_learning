import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter('logs')
image_path = '../dataset/hymenoptera_data/train/ants_image/6743948_2b8c096dda.jpg'
img = Image.open(image_path)
img_array = np.array(img)
writer.add_image('test', img_array,1,dataformats='HWC')
print(type(img_array))
print(img_array.shape)
# for i in range(100):
#     writer.add_scalar("y=x",i,i)

writer.close()