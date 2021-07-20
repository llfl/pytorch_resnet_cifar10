import torch
import torch.nn as nn
from rpc import RPConv

class rpcnet(nn.Module):
    def __init__(self, deploy=False):
        super(rpcnet,self).__init__()
        self.stem = nn.Conv2d(
            in_channels=3, 
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.deploy = deploy
        self.bn0 = nn.BatchNorm2d(64)

        self.layer1  = RPConv(64, 64 , deploy=self.deploy, scaling=False) 
        self.layer2  = RPConv(64, 64 , deploy=self.deploy, scaling=True)
        self.layer3  = RPConv(64, 128, deploy=self.deploy, scaling=False)
        self.layer4  = RPConv(128, 64, deploy=self.deploy, scaling=True) 
        self.layer5  = RPConv(64, 64,  deploy=self.deploy, scaling=False)
        self.layer6  = RPConv(64, 64 , deploy=self.deploy, scaling=True)
        self.layer7  = RPConv(64, 64,  deploy=self.deploy, scaling=False) 
        self.layer8  = RPConv(64, 32 , deploy=self.deploy, scaling=True)
        self.layer9  = RPConv(32, 32,  deploy=self.deploy, scaling=False)
        self.layer10 = RPConv(32, 10 , deploy=self.deploy, scaling=True)

    def forward(self, x):

        # stem
        x = self.act(self.bn0(self.stem(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        return x.view(x.size(0), -1)