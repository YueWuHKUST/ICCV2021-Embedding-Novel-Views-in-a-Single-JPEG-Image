import torch.nn as nn
def create_model(opt):
    from .framework import Network
    net = Network()

    net.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        net = nn.DataParallel(net, device_ids=opt.gpu_ids)
        return net
    else:
        return net
