import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from modules.mobile_netv3 import Block, SeModule
from modules.hlconv import *

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_dep(nn.Module):
    def __init__(self,ch_in,ch_out,use_hswish=False):
        super(conv_block_dep,self).__init__()
        hlConv = hlconv["dep_sep_conv"] 
        self.conv = nn.Sequential(
            hlConv(ch_in,  ch_out, use_hswish=use_hswish),
            hlConv(ch_out, ch_out, use_hswish=use_hswish),
        )
        
    def forward(self,x):
        x = self.conv(x)
        return x    
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(       
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x,size):
        x = F.interpolate(x,size=size,mode='trilinear',align_corners=True)  
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class UNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(UNet,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5,tuple(x4.shape[2:]))
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5,tuple(x3.shape[2:]))
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,tuple(x2.shape[2:]))
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,tuple(x1.shape[2:]))
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
    
class ISMNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,
                 num_feature_factor=0.6,
                 expansion = 1.5):
        super(ISMNet,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        
        nf_x1 = int(64 * num_feature_factor)
        nf_x1_ex = int(nf_x1 * expansion)
        self.Conv1 = nn.Sequential(Block(3, img_ch, nf_x1_ex, nf_x1, nn.ReLU(inplace=True), None, 1),
                                  )
        
        nf_x2 = int(128 * num_feature_factor)
        nf_x2_ex = int(nf_x2 * expansion)        
        self.Conv2 = nn.Sequential(Block(3, nf_x1, nf_x2_ex, nf_x2, nn.ReLU(inplace=True), None, 1),
                                  )
        
        nf_x3 = int(256 * num_feature_factor)
        nf_x3_ex = int(nf_x3 * expansion)          
        self.Conv3 = nn.Sequential(Block(3, nf_x2, nf_x3_ex, nf_x3, nn.ReLU(inplace=True), SeModule(nf_x3), 1),
                                  )        
        
        nf_x4 = int(512 * num_feature_factor)
        nf_x4_ex = int(nf_x4 * expansion)          
        self.Conv4 = nn.Sequential(Block(3, nf_x3, nf_x4_ex, nf_x4, nn.ReLU(inplace=True), SeModule(nf_x4), 1),
                                  )        

        nf_x5 = int(1024 * num_feature_factor)
        nf_x5_ex = int(nf_x5 * expansion)         
        self.Conv5 = nn.Sequential(Block(3, nf_x4, nf_x5_ex, nf_x5, nn.ReLU(inplace=True), SeModule(nf_x5), 1),
                                  )       
        
        nf_d5 = int(512 * num_feature_factor)
        self.Up5 = up_conv(ch_in=nf_x5, ch_out=nf_d5)
        self.Up_conv5 = conv_block_dep(ch_in=nf_d5 + nf_x4, ch_out=nf_d5)     
        
        nf_d4 = int(256 * num_feature_factor)
        self.Up4 = up_conv(ch_in=nf_d5, ch_out=nf_d4)
        self.Up_conv4 = conv_block_dep(ch_in=nf_d4 + nf_x3, ch_out=nf_d4) 
        
        nf_d3 = int(128 * num_feature_factor)
        self.Up3 = up_conv(ch_in=nf_d4, ch_out=nf_d3)
        self.Up_conv3 = conv_block_dep(ch_in=nf_d3 + nf_x2, ch_out=nf_d3) 
        
        nf_d2 = int(64 * num_feature_factor)
        self.Up2 = up_conv(ch_in=nf_d3, ch_out=nf_d2)
        self.Up_conv2 = conv_block_dep(ch_in=nf_d2 + nf_x1, ch_out=nf_d2)

        self.Conv_1x1 = nn.Conv3d(nf_d2, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)         
        
        # decoding + concat path
        d5 = self.Up5(x5,tuple(x4.shape[2:]))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,tuple(x3.shape[2:]))
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,tuple(x2.shape[2:]))
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,tuple(x1.shape[2:]))
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        return d1