import torch.nn as nn

class ConvClassifier(nn.Module):
    """
    Input shape: (batch_size, num_points, num_keys)
    permute to (batch_size, num_keys, num_points) and then apply 1D conv.
    """
    def __init__(self, num_keys, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_keys, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_points, num_keys)
        x = x.permute(0, 2, 1)  # (batch_size, num_keys, num_points)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = self.pool(x)   # -> (batch_size, 512, 1)
        x = x.view(x.size(0), -1)  # -> (batch_size, 512)

        x = self.drop1(self.relu4(self.fc1(x)))
        x = self.drop2(self.relu5(self.fc2(x)))
        out = self.fc3(x)  # -> (batch_size, num_classes)
        return out
