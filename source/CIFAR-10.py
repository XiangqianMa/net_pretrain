import numpy as np
import pickle
import os
import cv2


def unpickle(file):
    """
    CIFAR-10官网给出的原始数据是以python pickle的形式进行存储的，使用该函数对pickle文件进行解析
    :param file:文件名
    :return:Python字典
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def parse(file):
    """
    对单个文件进行解析
    :param file: 文件名
    :return:
    """
    dict = unpickle(file)
    data = dict[b'data']
    labels = dict[b'labels']
    data_row = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    image = data_row[41, :, :, :]
    image_ = cv2.resize(image, (448, 448))


if __name__ == '__main__':
    file = "F:\Chrome\download\cifar-10-python\cifar-10-batches-py"
    file_name = os.path.join(file, "data_batch_1")
    print(file_name)
    parse(file_name)


