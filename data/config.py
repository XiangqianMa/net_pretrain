IMAGENET_ROOT = ''

imagenet_config = {
    'lr_init': 1e-2,
    'lr_step': [5000, 9000],
    'wp_lr': 1e-3,
    'wp_step': 1000,
    'batch_size': 32,
    'max_iteration': 10000,
    'save_interval': 5000,
}