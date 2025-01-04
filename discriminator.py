import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # Output: (64, 128, 128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Output: (128, 64, 64)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # Output: (256, 32, 32)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # Output: (512, 16, 16)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)  # Output: (1024, 8, 8)

        # Fully connected layer to output a single probability for real/fake
        self.fc1 = nn.Linear(1024 * 8 * 8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.fc1(x)
        return torch.sigmoid(x)  # Probability of real/fake (0-1)