import torch
import torch.nn as nn
import rpcnet

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model =rpcnet.rpcnet()
    model.eval()

    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         nn.init.uniform_(module.running_mean, 0, 0.1)
    #         nn.init.uniform_(module.running_var, 0, 0.1)
    #         nn.init.uniform_(module.weight, 0, 0.1)
    #         nn.init.uniform_(module.bias, 0, 0.1)

    train_y = model(x)
    # for module in model.modules():
    #     if hasattr(module, 'switch_to_deploy'):
    #         module.switch_to_deploy()
    model = rpcnet.repvgg_model_convert(model)

    print(model)
    deploy_y = model(x)
    print('========================== The diff is')
    print(((train_y - deploy_y) ** 2).sum())