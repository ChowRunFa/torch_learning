from torchvision import transforms
# tensor 数据类型
from PIL import Image

img_path = '../dataset/hymenoptera_data/train/ants_image/0013035.jpg'
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)