import torch
import os
import sys
import argparse
import tqdm

from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

sys.path.append('.')
from models.mobilenet_bn import mobilenet_v2
from data import ImageNetVal
from data import config
from data import IMAGENET_VAL_ROOT
from data import image_val_config
from data import IMAGENET_VAL_ANN_ROOT

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Evaluate model.')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--dataset', default='imagenet', help='Dataset name.')
parser.add_argument('--model', default='mobilenet_bn', help='The model to be pretrained.')
parser.add_argument('--checkpoint', default='weights/mobilenet_imagenet_590000.pth', type=str, help='Checkpoint state_dict file.')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--multi_cuda', default=True, type=str2bool, help='Use multi gpus.')

args = parser.parse_args()

if not args.multi_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')

def eval():
    if args.dataset == 'imagenet':
        val_dataset = ImageNetVal(
            dataset_val_root=IMAGENET_VAL_ROOT,
            label_path=IMAGENET_VAL_ANN_ROOT,
            transforms=transforms.Compose(
                [
                    transforms.Resize([300, 300]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        )
    
    dataloader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=image_val_config['batch_size'],
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False
                        )
    
    # 定义网络结构
    if args.model == 'mobilenet_bn':
        net = mobilenet_v2(num_classes=image_val_config['num_classes'])
    net_gpus = net

    # 判断是否使用多gpus
    if args.cuda and args.multi_cuda:
        net_gpus = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print("GPU name is: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint))
    else:
        print("Please give your checkpoint file.")
    
    if args.cuda:
        print("Converting model to GPU...")
        net_gpus.cuda()
    
    net_gpus.eval()

    if args.cuda:
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
    else:
        correct = torch.zeros(1).squeeze()
        total = torch.zeros(1).squeeze()

    for images_batch, targets_batch in tqdm.tqdm(dataloader, desc="Evaluate model"):
        if args.cuda:
            images = Variable(images_batch.cuda())
            targets = Variable(targets_batch.cuda())
        else:
            images = Variable(images_batch)
            targets = Variable(targets_batch)

        predictions = net_gpus(images)
        predictions = nn.functional.log_softmax(predictions)
        predictions = torch.argmax(predictions, dim=1)
        targets = targets.squeeze(1)

        correct += (predictions == targets).sum().float()
        total += len(targets)
    
    print("{} images.".format(total.cpu().item()))
    accuracy = (correct / total).cpu().item()

    return accuracy

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    accuracy = eval()
    print("Accuracy: %.4f" % accuracy)
    pass

