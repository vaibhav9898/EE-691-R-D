import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import Config

def get_data_loaders(cfg):
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((cfg.mnist_mean,), (cfg.mnist_std,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((cfg.mnist_mean,), (cfg.mnist_std,))
    ])

    # Load datasets
    train_dataset = ImageFolder(root=cfg.train_data_dir, transform=train_transform)
    test_dataset = ImageFolder(root=cfg.test_data_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Print dataset info
    print("Training Dataset:")
    print(f"  Size: {len(train_dataset)}")
    print(f"  Shape: {next(iter(train_loader))[0].shape}")
    print(f"  Batch Size: {train_loader.batch_size}")
    print("\nTesting Dataset:")
    print(f"  Size: {len(test_dataset)}")
    print(f"  Shape: {next(iter(test_loader))[0].shape}")
    print(f"  Batch Size: {test_loader.batch_size}")

    return train_loader, test_loader