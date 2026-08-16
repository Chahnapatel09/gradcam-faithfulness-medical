from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_dataloaders(data_dir: str = "./data", batch_size: int = 32, num_workers: int = 2):
    """PneumoniaMNIST train/val/test loaders.

    Images are loaded at 224x224 and converted to 3-channel (as_rgb=True) to
    match the pretrained ResNet-18's expected input.
    """
    train_set = PneumoniaMNIST(split="train", transform=train_transform,
                                download=True, size=224, as_rgb=True, root=data_dir)
    val_set = PneumoniaMNIST(split="val", transform=eval_transform,
                              download=True, size=224, as_rgb=True, root=data_dir)
    test_set = PneumoniaMNIST(split="test", transform=eval_transform,
                               download=True, size=224, as_rgb=True, root=data_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
