import torch as t
import sys
sys.path.append("..")
import rpc 


class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight)

class QuanRPConv(rpc.RPConv):
    def __init__(self, m: rpc.RPConv, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == rpc.RPConv
        super().__init__(m.in_channels, m.out_channels,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         scaling=m.scaling,
                         deploy=m.deploy)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.quan_dense = t.nn.Sequential()
        self.quan_dense.add_module('LsqQuan',QuanConv2d(m.rpc_dense.conv, quan_w_fn=quan_w_fn, quan_a_fn=quan_a_fn))
        self.quan_dense.add_module('bn',m.rpc_dense.bn)

        self.quan_1x1 = t.nn.Sequential()
        self.quan_1x1.add_module('LsqQuan',QuanConv2d(m.rpc_1x1.conv, quan_w_fn=quan_w_fn, quan_a_fn=quan_a_fn))
        self.quan_1x1.add_module('bn',m.rpc_1x1.bn)

        if hasattr(m, 'rpc_identity'):
            self.rpc_identity = m.rpc_identity

        self.qdense_weight = t.nn.Parameter(m.rpc_dense.conv.weight.detach())
        self.q1x1_weight = t.nn.Parameter(m.rpc_1x1.conv.weight.detach())
        self.quan_w_fn.init_from(m.rpc_dense.conv.weight)

    def forward(self, x):

        # quantized_dense_weight = self.quan_w_fn(self.qdense_weight)
        # quantized_1x1_weight = self.quan_w_fn(self.q1x1_weight)
        quantized_act = self.quan_a_fn(x)
        if self.rpc_identity is None:
            identity = 0
        else:
            identity = self.rpc_identity(quantized_act)
        return self.quan_dense(quantized_act) + self.quan_1x1(quantized_act) + identity

class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)


QuanModuleMapping = {
    rpc.RPConv:  QuanRPConv,
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear,
}
