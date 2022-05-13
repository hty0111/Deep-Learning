import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2_1, c2_2, c3_1, c3_2, c4) -> None:
        super().__init__()
        # 1x1conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 1x1conv -> 3x3conv
        self.p2_1 = nn.Conv2d(in_channels, c2_1, kernel_size=1)
        self.p2_2 = nn.Conv2d(c2_1, c2_2, kernel_size=3, padding=1)
        # 1x1conv -> 5x5conv
        self.p3_1 = nn.Conv2d(in_channels, c3_1, kernel_size=1)
        self.p3_2 = nn.Conv2d(c3_1, c3_2, kernel_size=5, padding=2)
        # 3x3pool -> 1x1conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1024, num_classes))

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.fc(out)
        return out

if __name__=="__main__":
    x = torch.randn(32, 1, 96, 96)
    net = GoogLeNet()
    y = net(x)
    print(y.shape)
