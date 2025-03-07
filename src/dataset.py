import torch
from torch.utils.data import Dataset, DataLoader

class GazeDataset(Dataset):
    """
    PyTorch Dataset returning (feature, label) pairs,
    where feature has shape (num_points, num_keys)
    """
    def __init__(self, features, labels):
        # features: (N, num_points, num_keys)
        # labels:   (N,)
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(train_X, train_y, test_X, test_y, batch_size=32):
    """
    Create PyTorch DataLoaders from training and testing data.
    """
    train_dataset = GazeDataset(train_X, train_y)
    test_dataset = GazeDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader