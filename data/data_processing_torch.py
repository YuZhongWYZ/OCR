#%%
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import KFold

def load_and_preprocess_data():
    # Step 1: Load the MNIST dataset using PyTorch
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)

    # Combine train and test for K-fold splitting later
    x = torch.cat([mnist_dataset.data, mnist_test_dataset.data], dim=0).float()
    y = torch.cat([mnist_dataset.targets, mnist_test_dataset.targets], dim=0)

    # Step 2: Normalize the data
    # Min-Max normalization
    x_min_max = x / 255.0

    # Mean normalization
    mean = torch.mean(x, dim=(1, 2), keepdim=True)
    std = torch.std(x, dim=(1, 2), keepdim=True) + 1e-8
    x_mean = (x - mean) / std

    # Step 3: Prepare K-fold splitting
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store K-fold split indices for later use
    kf_splits = []
    for train_idx, val_idx in kf.split(x):
        kf_splits.append((train_idx.tolist(), val_idx.tolist()))

    return x_min_max, x_mean, y, kf_splits

if __name__ == "__main__":
    x_min_max, x_mean, y, kf_splits = load_and_preprocess_data()

    # Save processed data and splits to be reused in the main script
    torch.save({
        'x_min_max': x_min_max,
        'x_mean': x_mean,
        'y': y,
        'kf_splits': kf_splits
    }, "processed_data.pth")

    # Display summary
    print("Data normalization completed.")
    print(f"Min-Max normalized shape: {x_min_max.shape}")
    print(f"Mean normalized shape: {x_mean.shape}")
    print(f"Number of splits: {len(kf_splits)}")
