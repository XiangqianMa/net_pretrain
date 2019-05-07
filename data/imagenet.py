# Author: XiangqianMa
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import sys
import torch
import cv2

def one_hot(label, class_num):
    """返回label的one_hot编码
    Args:
        label: 样本的真实类标
        class_num: 类别总数
    """

    one_hot_encoding = torch.zeros(class_num)
    one_hot_encoding[label] = 1

    return one_hot_encoding

class ImageNet(Dataset):
    """加载ImageNet数据集

    ImageNet数据集的组织方式如下:
    dataset_root/class_0/xxx.jpg
    dataset_root/class_1/xxx.jpg
    ...
    ...

    Args:
        dataset_root:数据集的存放路径
        transform:对image施加的变换
    """

    def __init__(self, dataset_root, transform=None):

        self.dataset_root = dataset_root
        self.transform = transform

        # 得到真实类标及类标到id的映射
        self.classes, self.class_to_idx = self._get_classes()
        self.class_num = len(self.classes)

        # 得到所有样本对应的路径及其类别id
        self.samples = self._get_samples() # TODO


    def __getitem__(self, index): # TODO
        image_path = self.samples[index][0]
        class_label = self.samples[index][1]

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)
        
        label_one_hot = one_hot(class_label, self.class_num)
        
        return img, label_one_hot


    def __len__(self): # TODO
        samples_num = len(self.samples)
        return samples_num


    def _get_classes(self):
        """从数据集路径中依据folder名称解析出类别,并返回类别对应的id

        return:
            classes[list]:真实类别名称
            class_to_idx[dict]:真实类别到id的映射
        """
        classes = [d.name for d in os.scandir(self.dataset_root) if d.is_dir()]
        classes.sort()

        class_to_idx = {classes[idx]:idx for idx in range(len(classes))}

        return classes, class_to_idx


    def _get_samples(self):
        """得到所有样本的路径以及样本对应的类标id

        return:
            samples[list]:包含的元素为list,每一个list包含样本路径和类标id
        """
        samples = list()
        folders = os.listdir(self.dataset_root)
        # 对每一个文件夹下的样本进行处理
        for class_folder in folders:
            class_folder_path = os.path.join(self.dataset_root, class_folder)
            for sample_name in os.listdir(class_folder_path):
                sample_path = os.path.join(class_folder_path, sample_name)
                # 得到一个样本及其对应的类别id
                sample = [sample_path, self.class_to_idx[class_folder]]
                samples.append(sample)
        
        return samples


if __name__ == "__main__":
    root = "/home/mxq/Project/data/ImageNet/train"

    dataset = ImageNet(root, 
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                    ))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
        )

    batch_iterator = iter(dataloader)
    img, label = next(batch_iterator)
    
    print(np.shape(img))
    print(label)
    pass