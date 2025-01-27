import os
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Automatically determine the number of workers based on CPU count
NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        data_dir: Path,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    # Load the dataset from the directory
    full_data = datasets.ImageFolder(data_dir, transform=transform)

    # Split the data into 80% training and 20% testing
    train_len = int(len(full_data) * 0.8)
    test_len = len(full_data) - train_len
    train_set, test_set = torch.utils.data.random_split(full_data, [train_len, test_len])

    # Get class names
    class_names = full_data.classes

    # Create DataLoaders for training and testing
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
