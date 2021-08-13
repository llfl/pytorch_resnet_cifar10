import torch
import torch.nn as nn
from rpc import RPConv
import copy

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
        self.layer2  = RPConv(64, 64 , deploy=self.deploy, scaling=False)
        self.layer3  = RPConv(64, 128, deploy=self.deploy, scaling=False)
        self.layer4  = RPConv(128, 64, deploy=self.deploy, scaling=False) 
        self.layer5  = RPConv(64, 64,  deploy=self.deploy, scaling=False)
        self.layer6  = RPConv(64, 64 , deploy=self.deploy, scaling=False)
        self.layer7  = RPConv(64, 64,  deploy=self.deploy, scaling=False) 
        self.layer8  = RPConv(64, 32 , deploy=self.deploy, scaling=False)
        self.layer9  = RPConv(32, 32,  deploy=self.deploy, scaling=False)
        self.layer10 = RPConv(32, 10 , deploy=self.deploy, scaling=False)

        self.pooling = nn.MaxPool2d(2)
        self.act = nn.ReLU()

    def forward(self, x):

        # stem
        x = self.act(self.bn0(self.stem(x)))

        x = self.layer1(x)
        x = self.pooling(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pooling(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pooling(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pooling(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.pooling(x)
        x = self.layer10(x)

        return x.view(x.size(0), -1)

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model