import os.path

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.imgs = os.listdir(self.path)


    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir

        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root_dir = '../dataset/hymenoptera_data/train'
    ants_label_dir = 'ants_image'
    bees_label_dir = 'bees_image'
    ants_dataset = MyDataset(root_dir,ants_label_dir)
    bees_dataset = MyDataset(root_dir,bees_label_dir)
    train_dataset = ants_dataset + bees_dataset
    print(len(train_dataset))