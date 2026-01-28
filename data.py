"""
Data loaders for MNIST and CIFAR-10.
Returns train and validation dataloaders.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset_name: str, batch_size: int = 128, val_split=0.1):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    # split train -> train + val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
