from model import common

import torch.nn as nn
import torch.nn.functional as F
import torch

url = {

}


def make_model(args, parent=False):
    return SISR_Model(args)


class SISR_Model(nn.Module):
    def __init__(self, args):
        super(SISR_Model, self).__init__()

        scale = self.scale

        self.upsample = common.UpsampleBlock(n_channels=64, scale=scale, multi_scale=len(scale), group=1)

    def forward(self, x, scale):
        out = self.upsample(x, scale)

        return out

    def load_state_dict(self, state_dict, strict=True):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
