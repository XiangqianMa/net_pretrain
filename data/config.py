IMAGENET_ROOT = "/home/apple/data/MXQ/imagenet/ILSVRC2015/Data/CLS-LOC/train"
IMAGENET_VAL_ROOT = "/home/apple/data/MXQ/imagenet/ILSVRC2015/Data/CLS-LOC/val"
IMAGENET_VAL_ANN_ROOT = "/home/apple/data/MXQ/imagenet/ILSVRC2015/Annotations/CLS-LOC/val"

imagenet_config = {
    'num_classes': 1000,
    'lr_init': 0.1,
    'gamma': 0.1,
    'lr_interval': 30,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'batch_size': 64,
    'max_epoch': 90,
}

image_val_config = {
    'num_classes': 1000, 
    'batch_size': 80,
}