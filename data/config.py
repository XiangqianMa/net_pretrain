IMAGENET_ROOT = "/home/apple/data/MXQ/imagenet/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train"
IMAGENET_VAL_ROOT = "/home/apple/data/MXQ/imagenet/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val"

imagenet_config = {
    'num_classes': 2,
    'lr_init': 0.045,
    'lr_step': [140000, 280000, 420000, 560000],
    'wp_lr': 1e-3,
    'wp_step': 100,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'batch_size': 32,
    'max_iteration': 600000,
    'save_interval': 10000,
}

image_val_config = {
    'num_classes': 1000, 
    'batch_size': 32,
}