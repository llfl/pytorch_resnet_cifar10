import torch
import torch.nn as nn

class lightconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(lightconv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class lightnet(nn.Module):
    def __init__(self):
        super(lightnet, self).__init__()
        self.stem = nn.Conv2d(
            in_channels=3, 
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn0 = nn.BatchNorm2d(64)

        self.layer1  = lightconv(64, 64) 
        self.layer2  = lightconv(64, 64)
        self.layer3  = lightconv(64, 128)
        self.layer4  = lightconv(128, 64) 
        self.layer5  = lightconv(64, 64)
        self.layer6  = lightconv(64, 64)
        self.layer7  = lightconv(64, 64) 
        self.layer8  = lightconv(64, 32)
        self.layer9  = lightconv(32, 32)
        self.layer10 = lightconv(32, 10)

        self.pooling = nn.MaxPool2d(2)
        self.act = nn.ReLU()

    def forward(self, x):

        # stem
        x = self.act(self.bn0(self.stem(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pooling(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pooling(x)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.pooling(x)

        x = self.layer9(x)
        x = self.layer10(x)
        x = self.pooling(x)

        return x.view(x.size(0), -1)