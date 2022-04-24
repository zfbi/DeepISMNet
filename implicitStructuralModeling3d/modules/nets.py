import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .mobile_netv3 import MobileNetV3_Encoder as MobileNetV3
from .mobile_netv3 import MobileNetV3_EncoderHighlight as MobileNetV32
from .hlconv import *

class SimpleDecoder(nn.Module):
    def __init__(self, ch_in, ch_out, ch_low=None, conv_operator='dep_sep_conv', kernel_size=3, 
                 batch_norm=nn.BatchNorm3d, use_hswish=False):
        super(SimpleDecoder, self).__init__()
        
        self.mode = "trilinear"
        
        if ch_low is None:
            ch_low = ch_in

        hlConv3d = hlconv[conv_operator]
        self.conv1 = hlConv3d(ch_in, ch_out, kernel_size, 1, batch_norm, use_hswish=use_hswish)
        self.conv2 = hlConv3d(ch_low, ch_out, kernel_size, 1, batch_norm, use_hswish=use_hswish)       

    def forward(self, l_encode, l_low):
        l_encode = self.conv1(l_encode)
        l_encode = F.interpolate(l_encode, size=tuple(l_low.shape[2:]), mode=self.mode, align_corners=True)  
        l_cat = torch.cat((l_encode, l_low), dim=1)    
        return self.conv2(l_cat)
    


def OutputLayer(ch_in, ch_out, conv_operator='std_conv', kernel_size=3, batch_norm=nn.BatchNorm3d):
    return nn.Sequential(
        hlconv[conv_operator](ch_in, ch_out, kernel_size, 1, batch_norm),
        nn.Conv3d(ch_out, ch_out, kernel_size, 1, padding=kernel_size//2, bias=False)
    )    

class IGMNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1, nfts=1.4):
        super(IGMNet,self).__init__()
        
        self.backbone = MobileNetV3(img_ch=img_ch, nfts=nfts)
        
        ch_in, ch_out = int(160*nfts), int(160*nfts)
        self.decoder_layer4 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)
        
        ch_in, ch_out = int(160*nfts), int(40*nfts)
        self.decoder_layer3 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)
        
        ch_in, ch_out = int(40*nfts), int(24*nfts)
        self.decoder_layer2 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)
        
        ch_in, ch_out = int(24*nfts), int(16*nfts)
        self.decoder_layer1 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)        

        self.output_layer = OutputLayer(int(16*nfts), output_ch)

    def forward(self,x):
        # encoding path      
        
        x1, x2, x3, x4, x5 = self.backbone(x)
            
        # decoding + concat path  
        d4 = self.decoder_layer4(x5, x4)
        d3 = self.decoder_layer3(d4, x3)
        d2 = self.decoder_layer2(d3, x2)
        d1 = self.decoder_layer1(d2, x1)
        
        d0 = self.output_layer(d1)

        return d0
    
class IGMNetPlus(nn.Module):
    def __init__(self,img_ch=1,output_ch=1, nfts=1.3):
        super(IGMNetPlus,self).__init__()
        
        self.backbone = MobileNetV32(img_ch=img_ch, nfts=nfts)
        
        ch_in, ch_out = int(160*nfts), int(160*nfts)
        self.decoder_layer4 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)
        
        ch_in, ch_out = int(160*nfts), int(40*nfts)
        self.decoder_layer3 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)
        
        ch_in, ch_out = int(40*nfts), int(24*nfts)
        self.decoder_layer2 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)
        
        ch_in, ch_out = int(24*nfts), int(16*nfts)
        self.decoder_layer1 = SimpleDecoder(ch_in=ch_in, ch_out=ch_out, ch_low=ch_out*2, use_hswish=True)        

        self.output_layer = OutputLayer(int(16*nfts), output_ch)

    def forward(self,x):
        # encoding path      
        
        x1, x2, x3, x4, x5 = self.backbone(x)
            
        # decoding + concat path  
        d4 = self.decoder_layer4(x5, x4)
        d3 = self.decoder_layer3(d4, x3)
        d2 = self.decoder_layer2(d3, x2)
        d1 = self.decoder_layer1(d2, x1)
        
        d0 = self.output_layer(d1)

        return d0