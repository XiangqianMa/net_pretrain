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
import time
import xml.etree.ElementTree as ET
import pickle


sys.path.append(".")
from data.config import imagenet_config

def one_hot(label, class_num):
    """返回label的one_hot编码
    Args:
        label: 样本的真实类标
        class_num: 类别总数
    """

    one_hot_encoding = torch.zeros(class_num)
    one_hot_encoding[label] = 1
    one_hot_encoding = one_hot_encoding.long()

    return one_hot_encoding

def parse_xml(xml_path):
    """解析xml文件，得到样本的类标
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    name_tmp = next(root.iter('object')).find('name').text
    for obj in root.iter('object'):
        name = obj.find('name').text 
        if name_tmp == name:
            continue
        else:
            print("There are different labels in {}".format(xml_path))
            break

    return name

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
        super(ImageNet, self).__init__()
        self.dataset_root = dataset_root
        self.transform = transform

        # 得到真实类标及类标到id的映射
        self.classes, self.class_to_idx = self._get_classes()
        self.class_num = len(self.classes)

        # 得到所有样本对应的路径及其类别id
        self.samples = self._get_samples() # TODO

        # 将类别到索引的字典存储为pickle
        if not os.path.exists("./data/imagenet_class_idx.pkl"):
            print("Creating imagenet_class_idx.pkl...")
            with open("./data/imagenet_class_idx.pkl", "wb") as f:
                pickle.dump(self.class_to_idx, f, -1)
        

    def __getitem__(self, index):
        image_path = self.samples[index][0]
        class_label = self.samples[index][1]

        img = Image.open(image_path)
        img = img.convert('RGB')
     
        if self.transform:
            img = self.transform(img)

        label = torch.tensor([class_label])
        return img, label


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


class ImageNetVal(Dataset):
    def __init__(self, dataset_val_root, label_path, transforms=None):
        super(ImageNetVal, self).__init__()
        self.dataset_val_root = dataset_val_root
        self.label_path = label_path
        self.transforms = transforms

        with open("./data/imagenet_class_idx.pkl", 'rb') as f:
            self.class_to_idx = pickle.load(f)
        self.class_num = len(self.class_to_idx)
        self.samples = self._get_samples()
    

    def __getitem__(self, index):
        image_path = self.samples[index][0]
        class_label = self.samples[index][1]

        img = Image.open(image_path)
        img = img.convert('RGB')
     
        if self.transforms:
            img = self.transforms(img)
        
        target = torch.tensor([class_label])

        return img, target
    

    def __len__(self):
        return len(self.samples)


    def _get_samples(self):
        """得到所有样本的路径及对应的类别索引
        """
        samples = list()
        for sample_name in os.listdir(self.dataset_val_root):
            # 样本路径
            sample_path = os.path.join(self.dataset_val_root, sample_name)
            # 得到类标
            annotation_name = sample_name.split('.')[0] + '.xml'
            annotation_path = os.path.join(self.label_path, annotation_name)
            class_name = parse_xml(annotation_path)

            target = self.class_to_idx[class_name]
            sample = [sample_path, target]
            samples.append(sample)
        
        return samples




if __name__ == "__main__":
    root = "/home/apple/data/MXQ/imagenet/ILSVRC2015/Data/CLS-LOC/train"
    val_annot_root = "/home/apple/data/MXQ/imagenet/ILSVRC2015/Annotations/CLS-LOC/val"
    val_root = "/home/apple/data/MXQ/imagenet/ILSVRC2015/Data/CLS-LOC/val"

    dataset = ImageNet(root, 
                    transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=imagenet_config['mean'], std=imagenet_config['std']),
                    ]
                    ))

    val_dataset = ImageNetVal(
        dataset_val_root = val_root,
        label_path = val_annot_root,
        transforms=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        # num_workers=1,
        pin_memory=True
        )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    batch_iterator = iter(dataloader)
    img, label = next(batch_iterator)
    img = img[0, :, :, :]
    img = img.permute(1 , 2, 0)
    cv2.imshow("win", img.numpy()[:, :, (2, 1, 0)])
    cv2.waitKey(0)
    print(np.shape(img))
    print(np.shape(label))
    
    val_batch_iterator = iter(val_dataloader)
    val_img, val_label = next(val_batch_iterator)
    img = val_img[0]
    img = img.permute(1, 2, 0)
    cv2.imshow("win", img.numpy()[:, :, [2, 1, 0]])
    cv2.waitKey(0)
    print(val_label.shape)

    # for annotation in os.listdir(val_annot_root):
    #     annotation_path = os.path.join(val_annot_root, annotation)
    #     name = parse_xml(annotation_path)
    #     print(name)

    pass