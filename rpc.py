import torch.nn as nn
import numpy as np
import torch

def conv_bn(in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1):
    result = nn.Sequential()
    if padding:
        result.add_module('padding', nn.ZeroPad2d(padding))
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RPConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=None,dilation=1, groups=1, kernel_size=3, deploy=False, scaling=False, custom=False):
        super(RPConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scaling = scaling

        assert kernel_size == 3 or kernel_size == 1
        self.kernel_size = kernel_size
        self.dilation = dilation
        

        if self.scaling:
            self.stride = 2
            self.padding = (0,1,0,1)
            self.padding11 = 0
        else:
            self.stride = 1
            self.padding = 1
            self.padding11 = 0
        if custom:
            self.stride = stride
            self.padding = padding

        #   Considering dilation, the actuall size of rbr_dense is  kernel_size + 2*(dilation - 1)
        #   For the same output size:     (padding - padding_11) ==  (kernel_size + 2*(dilation - 1) - 1) // 2
        # padding_11 = padding - (kernel_size + 2*(dilation - 1) - 1) // 2
        # assert padding_11 >= 0, 'It seems that your configuration of kernelsize (k), padding (p) and dilation (d) will ' \
        #                         'reduce the output size. In this case, you should crop the input of conv1x1. ' \
        #                         'Since this is not a common case, we do not consider it. But it is easy to implement (e.g., self.rbr_1x1(inputs[:,:,1:-1,1:-1])). ' \
        #                         'The common combinations are (k=3,p=1,d=1) (no dilation), (k=3,p=2,d=2) and (k=3,p=4,d=4) (PSPNet).'
        
        self.nonlinearity = nn.ReLU()

        
        if deploy:
            self.reparam_block = nn.Sequential()
            self.reparam_block.add_module('padding', nn.ZeroPad2d(self.padding))
            self.reparam_block.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=kernel_size, stride=self.stride, dilation=dilation, groups=groups, bias=False))

        else:
            self.rpc_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and self.stride == 1 else None
            self.rpc_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=self.stride, padding=self.padding, dilation=dilation, groups=groups)
            self.rpc_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride, padding=self.padding11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'reparam_block'):
            return self.nonlinearity(self.reparam_block(inputs))
        
        if self.rpc_identity is None:
            id_out = 0
        else:
            id_out = self.rpc_identity(inputs)
        return self.nonlinearity(self.rpc_dense(inputs) + self.rpc_1x1(inputs) + id_out)
        

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rpc_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rpc_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rpc_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_block'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_block = nn.Sequential()
        self.reparam_block.add_module('padding', nn.ZeroPad2d(self.padding))
        self.reparam_block.conv = nn.Conv2d(in_channels=self.rpc_dense.conv.in_channels, 
                                     out_channels=self.rpc_dense.conv.out_channels,
                                     kernel_size=self.rpc_dense.conv.kernel_size, stride=self.rpc_dense.conv.stride,
                                     padding=self.rpc_dense.conv.padding, dilation=self.rpc_dense.conv.dilation, groups=self.rpc_dense.conv.groups, bias=True)
        self.reparam_block.conv.weight.data = kernel
        self.reparam_block.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rpc_dense')
        self.__delattr__('rpc_1x1')
        if hasattr(self, 'rpc_identity'):
            self.__delattr__('rpc_identity')

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model