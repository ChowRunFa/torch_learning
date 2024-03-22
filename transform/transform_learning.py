from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = '../dataset/hymenoptera_data/train/ants_image/0013035.jpg'

def tensorboard_transform():

    # tensor 数据类型


    img = Image.open(img_path)

    writer = SummaryWriter('logs')

    tensor_trans = transforms.ToTensor()
    tensor_img = tensor_trans(img)

    writer.add_image('Tensor Image', tensor_img)

    writer.close()

def useful_transform():
    img = Image.open(img_path)

    writer = SummaryWriter('logs')

    trans_tensor = transforms.ToTensor()
    img_tensor = trans_tensor(img)

    writer.add_image('Tensor', img_tensor)

    trans_norm = transforms.Normalize([0.3, 0.1, 0.5],[2,1,0.5])
    img_norm = trans_norm(img_tensor)
    print(img_norm[0][0][0])

    writer.add_image('Normalized Image', img_norm)
    writer.close()

useful_transform()