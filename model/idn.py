from model import common

import torch.nn as nn
import torch.nn.functional as F
import torch

url = {

}


def make_model(args, parent=False):
    return IDN(args, num_features=64, d=16, s=4)


class FBlock(nn.Module):
    def __init__(self, num_features):
        super(FBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        return self.module(x)


class DBlock(nn.Module):
    def __init__(self, num_features, d, s):
        super(DBlock, self).__init__()
        self.num_features = num_features
        self.s = s
        self.enhancement_top = nn.Sequential(
            nn.Conv2d(num_features, num_features - d, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - d, num_features - 2 * d, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - 2 * d, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )
        self.enhancement_bottom = nn.Sequential(
            nn.Conv2d(num_features - d, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features, num_features - d, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - d, num_features + d, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )
        self.compression = nn.Conv2d(num_features + d, num_features, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.enhancement_top(x)
        slice_1 = x[:, :int((self.num_features - self.num_features / self.s)), :, :]
        slice_2 = x[:, int((self.num_features - self.num_features / self.s)):, :, :]
        x = self.enhancement_bottom(slice_1)
        x = x + torch.cat((residual, slice_2), 1)
        x = self.compression(x)
        return x


class DeConv(nn.Module):
    def __init__(self, num_features, scale):
        super(DeConv, self).__init__()

        if len(scale) == 3:
            self.multi_scale = True

            self.deconv1 = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=scale[0], padding=8,
                                              output_padding=1)
            self.deconv2 = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=scale[1], padding=8,
                                              output_padding=1)
            self.deconv3 = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=scale[2], padding=8,
                                              output_padding=1)
        elif len(scale) == 2:
            self.multi_scale = True

            self.deconv1 = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=scale[0], padding=8,
                                              output_padding=1)
            self.deconv2 = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=scale[1], padding=8,
                                              output_padding=1)
        else:
            self.multi_scale = False
            self.deconv = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=scale[0], padding=8,
                                             output_padding=1)

    def forward(self, x, idx_scale, outputsize):
        if self.multi_scale:
            if idx_scale == 0:
                return self.deconv1(x, output_size=outputsize)
            elif idx_scale == 1:
                return self.deconv2(x, output_size=outputsize)
            elif idx_scale == 2:
                return self.deconv3(x, output_size=outputsize)
        else:
            return self.deconv(x, output_size=outputsize)


class IDN(nn.Module):
    def __init__(self, args, num_features, d, s):
        super(IDN, self).__init__()
        self.scale = args.scale
        self.idx_scale = 0
        num_features = num_features
        d = d
        s = s

        self.fblock = FBlock(num_features)
        self.dblocks = nn.Sequential(*[DBlock(num_features, d, s) for _ in range(4)])
        self.deconv = DeConv(num_features, self.scale)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.sub_mean(x)

        if len(self.scale) > 1:
            bicubic = F.interpolate(x, scale_factor=self.scale[self.idx_scale], mode='bicubic', align_corners=False)
        else:
            bicubic = F.interpolate(x, scale_factor=self.scale[0], mode='bicubic', align_corners=False)

        x = self.fblock(x)
        x = self.dblocks(x)
        x = self.deconv(x, self.idx_scale, bicubic.size())

        out = x + bicubic
        out = self.add_mean(out)
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

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
