import torch

def adjust_learning_rate(optimizer, lr_init, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr_init * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warming_up_learning_rate(optimizer, wp_lr):
    for para_group in optimizer.param_groups:
        para_group['lr'] = wp_lr