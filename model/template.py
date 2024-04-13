from model import common

import torch.nn as nn
import torch.nn.functional as F
import torch

# if you have pretrain weights, write url here
url = {

}


# Add params such as num_features.
# This is one of base method.
def make_model(args, parent=False):
    return SISR(args, num_features=64)


# This is your SISR model.
class SISR(nn.Module):
    def __init__(self, args, num_features):
        super(SISR, self).__init__()
        self.scale = args.scale
        self.idx_scale = 0
        num_features = num_features

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self._initialize_weights()

    # This is initialize weights method.
    # This is a base method.
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # This is your model's forward method.
    def forward(self, x):
        x = self.sub_mean(x)
        out = self.add_mean(x)
        return out

    # This is load state method.
    # This is a base method.
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

    # if you use the idx_scale in your model, please add the method.
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
