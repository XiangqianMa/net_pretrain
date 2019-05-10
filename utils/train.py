import torch
import argparse
import os

from torch import nn
from torchvision import transforms
from torch.autograd import Variable 
import torch.backends.cudnn as cudnn
import torch.optim as optim

import models
from loss import classification_loss
from data import IMAGENET_ROOT, imagenet_config
from data import ImageNet
from utils import adjust_learning_rate, warming_up_learning_rate

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Pretrain model with imagenet dataset.')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--dataset', default='imagenet', help='Dataset name.')
parser.add_argument('--num_classes', default=1000, type=int, help='Dataset name.')
parser.add_argument('--dataset_root', default=IMAGENET_ROOT, help='Dataset root directory path')
parser.add_argument('--model', default='mobilenet_bn', help='The model to be pretrained.')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=0, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--multi_cuda', default=True, type=str2bool, help='Use multi gpus.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')

args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    # 定义数据集
    dataset_root = args.dataset_root
    if args.dataset == 'imagenet':
        dataset = ImageNet(
                        dataset_root, 
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ])
                        )
    
    dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True
                            )

    # 定义网络结构
    if args.model == 'mobilenet_bn':
        net = models.mobilenet_v2(args.num_classes)
    net_gpus = net

    # 判断是否使用多gpus
    if args.cuda and args.multi_cuda:
        net_gpus = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 初始化网络权重, 这里要使用net
    if args.resume:
        net.load_state_dict(torch.load(args.resume))
    else:
        net.init_weights()

    # 将net转移至gpu中
    if args.cuda:
        net_gpus.cuda()
    
    net_gpus.train()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=imagenet_config['lr_init'], momentum=args.momentum,
                            weight_decay=args.weight_decay)
    criterion = classification_loss()

    max_iteration = imagenet_config['max_iteration']
    
    batch_iterator = next(dataloader)
    
    step_index = 0
    for iteration in range(args.start_iter, max_iteration):

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(batch_iterator)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = Variable(targets.cuda())
        else:
            images = Variable(images)
            targets = Variable(targets)

        predictions = net_gpus(images)

        # 调整学习率，采用warming_up机制
        if iteration == 0:
            print('-------------l_r is {}.-----------------------\n'.format(args.wp_lr))
            warming_up_learning_rate(optimizer, imagenet_config['wp_lr'])
        elif iteration == imagenet_config['wp_step']:
            print('-------------iter: {}: Adjust l_r to: {}-------------\n'.format(iteration, args.lr*(args.gamma**0)))
            adjust_learning_rate(optimizer, imagenet_config['lr_init'], args.gamma, 0)
        elif iteration in imagenet_config['lr_step']:
            step_index += 1
            print('-------------iter: {}: Adjust l_r to: {}-------------\n'.format(iteration, args.lr*(args.gamma**step_index)))
            adjust_learning_rate(optimizer, imagenet_config['lr_init'], args.gamma, step_index)

        optimizer.zero_grad()
        model_loss = criterion(targets, predictions)
        model_loss.backward()
        optimizer.step()

        if iteration%10 == 0:
            print("Loss: {}".format(model_loss.item()))

if __name__ == "__main__":
    train()
    pass


    


