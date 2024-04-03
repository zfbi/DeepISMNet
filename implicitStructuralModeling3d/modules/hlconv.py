import torch
import torch.nn as nn
import torch.nn.functional as F

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

def conv_bn(inp, oup, k=3, s=1, batch_norm=nn.BatchNorm3d, use_hswish=False):
    
    if use_hswish:
        activation = hswish()        
    else:
        activation = nn.ReLU(inplace=True)
        
    return nn.Sequential(
        nn.Conv3d(inp, oup, k, s, padding=k//2, bias=False),
        batch_norm(oup),
        activation,
    )    

def dep_sep_conv_bn(inp, oup, k=3, s=1, batch_norm=nn.BatchNorm3d, use_hswish=False):
    
    if use_hswish:
        activation = hswish()        
    else:
        activation = nn.ReLU(inplace=True)    
    
    return nn.Sequential(
        nn.Conv3d(inp, inp, k, s, padding=k//2, groups=inp, bias=False),
        batch_norm(inp),
        activation,
        nn.Conv3d(inp, oup, 1, 1, padding=0, bias=False),
        batch_norm(oup),
        activation,
    )

hlconv = {
    'std_conv': conv_bn,
    'dep_sep_conv': dep_sep_conv_bn
}