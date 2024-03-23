

## 自定义 PyTorch 数据集类

本示例展示了如何创建一个自定义的 PyTorch `Dataset` 类，用于处理图像数据集，其中每个类别的图像存放在以该类别命名的子文件夹中。

### 导入必要的库

```python
import os.path
from torch.utils.data import Dataset
from PIL import Image
```

- `os.path`: 用于文件路径操作。
- `Dataset`: PyTorch 数据集的基类。
- `Image`: 来自 PIL 库，用于打开、操作和保存多种不同格式的图像。

### 定义自定义数据集类 `MyDataset`

```python
class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.imgs = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.imgs)
```

- `__init__`: 构造函数接收根目录和标签目录，然后生成图像的完整路径列表。
- `__getitem__`: 根据索引返回一对 `(img, label)`，其中 `img` 是 PIL 图像对象，`label` 是图像的类别名称。
- `__len__`: 返回数据集中图像的数量。

### 实例化并组合数据集

```python
if __name__ == '__main__':
    root_dir = '../dataset/hymenoptera_data/train'
    ants_label_dir = 'ants_image'
    bees_label_dir = 'bees_image'
    ants_dataset = MyDataset(root_dir, ants_label_dir)
    bees_dataset = MyDataset(root_dir, bees_label_dir)
    train_dataset = ants_dataset + bees_dataset  # 这里需要修改，因为不能直接相加
    print(len(train_dataset))
```

在主函数中，我们创建了两个 `MyDataset` 实例，分别对应蚂蚁和蜜蜂的图像。然后尝试通过 `+` 运算符合并这两个数据集