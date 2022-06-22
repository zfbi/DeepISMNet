import torch.nn as nn
import torch

from functools import reduce

class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class Discriminator(nn.Module):
    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()

        in_channels = 1
        out_channels = 64
        depth = 7
        bn = not gan_type == 'WGAN_GP'
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)
        
        patch_size = [k // (2 ** ((depth + 1) // 2)) for k in args['inshape']]
        
        m_classifier = [
            nn.Linear(out_channels * reduce(lambda x,y:x*y, patch_size), 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        
        output = self.classifier(features.view(features.size(0), -1))

        return output