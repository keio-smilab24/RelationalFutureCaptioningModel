from curses import KEY_ENTER
import torch
import torch.nn as nn
from torchinfo import summary

class CNN(nn.Module):
    """
    次元数調整用のCNN Module
    """
    def __init__(self, num_channels: int=3):
        super().__init__()

        self.cnn1 = nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn2 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten(start_dim=1)

        self.linear = nn.Linear(3136, 768)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor) : (B, 3*224*224)
        """
        B, L, D = x.size()
        x = x.view(-1, 3, 224, 224)

        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.linear(x)

        x = x.view(B, L, -1)

        return x

if __name__ == "__main__":
    model = CNN()

    inputs = torch.randn(size=(16, 3*224*224))
    print(inputs.shape)
    summary(model, inputs.size())
