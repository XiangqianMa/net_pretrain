IMAGENET_ROOT = "/media/lab3/D78F4730A17C1FD2/mxq/datasets/imageNet/ILSVRC2015/Data/CLS-LOC/train"

imagenet_config = {
    'num_classes': 1000,
    'lr_init': 1e-2,
    'lr_step': [40000, 120000, 160000, 180000],
    'wp_lr': 1e-3,
    'wp_step': 1000,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'batch_size': 64,
    'max_iteration': 600000,
    'save_interval': 5000,
}