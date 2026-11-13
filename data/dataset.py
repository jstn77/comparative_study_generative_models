import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from configurations.configuration import (DATA_DIR, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

def get_data_loaders():
    #transform (preprocess)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #download
    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    full_dataset = torch.utils.data.ConcatDataset([
        train_dataset,
        test_dataset
    ])

    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = int(total_size * TEST_RATIO)

    #splitting dataset
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    #data loader
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nDataset successfully loaded!")
    print("Total dataset:", total_size)
    print("Training:", len(train_set))
    print("Validation:", len(val_set))
    print("Testing:", len(test_set))

    return train_loader, val_loader, test_loader