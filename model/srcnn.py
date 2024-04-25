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
    return SRCNN(args)


# This is your SISR model.
class SRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRCNN, self).__init__()
        self.scale = args.scale
        self.idx_scale = 0

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        self.body = nn.Sequential(
            conv(args.n_colors, args.n_feats * 2, 9),
            nn.ReLU(inplace=True),
            conv(args.n_feats * 2, args.n_feats, 5),
            nn.ReLU(inplace=True),
            conv(args.n_feats, args.n_colors, 5)
        )

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


    # This is your model's forward method.
    def forward(self, x):
        x = self.sub_mean(x)

        x = self.body(x)

        x = self.add_mean(x)
        
        return x

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
