'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__() 
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        
        self.se = semodule
        
        self.conv1 = nn.Conv3d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(expand_size)
        self.nolinear1 = nolinear
        
        self.conv2 = nn.Conv3d(expand_size, expand_size, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm3d(expand_size)
        self.nolinear2 = nolinear
        
        self.conv3 = nn.Conv3d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv3d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool3d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv3d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool3d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out
    
# zfbi
class MobileNetV3_Encoder(nn.Module):
    def __init__(self, img_ch=1, nfts=1):
        super(MobileNetV3_Encoder, self).__init__()
        
        ch_in, ch_out = img_ch, int(16*nfts)
        self.block0 = nn.Sequential( 
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ch_out),
            hswish(),                       
            Block(3, ch_out, ch_out*1, ch_out, nn.ReLU(inplace=True), None, 1),
        )        
        
        ch_in, ch_out = int(16*nfts), int(24*nfts)
        self.block1 = nn.Sequential(
            Block(3, ch_in, ch_in*4, ch_out, nn.ReLU(inplace=True), None, 2),
            Block(3, ch_out, ch_out*3, ch_out, nn.ReLU(inplace=True), None, 1),
        )
        
        ch_in, ch_out = int(24*nfts), int(40*nfts)
        self.block2 = nn.Sequential(
            Block(5, ch_in, ch_in*3, ch_out, nn.ReLU(inplace=True), SeModule(ch_out), 2),
            Block(5, ch_out, ch_out*3, ch_out, nn.ReLU(inplace=True), SeModule(ch_out), 1),
            Block(5, ch_out, ch_out*3, ch_out, nn.ReLU(inplace=True), SeModule(ch_out), 1),
        )
        
        ch_in, ch_out0, ch_out1, ch_out = int(40*nfts), int(80*nfts), int(112*nfts), int(160*nfts)
        self.block3 = nn.Sequential(
            Block(3, ch_in, ch_in*6, ch_out0, hswish(), None, 2),
            Block(3, ch_out0, ch_out0*2, ch_out0, hswish(), None, 1), # 2.5
            Block(3, ch_out0, ch_out0*2, ch_out0, hswish(), None, 1), # 2.3
            Block(3, ch_out0, ch_out0*2, ch_out0, hswish(), None, 1), # 2.3
            Block(3, ch_out0, ch_out0*6, ch_out1, hswish(), SeModule(ch_out1), 1),
            Block(3, ch_out1, ch_out1*6, ch_out1, hswish(), SeModule(ch_out1), 1),
            Block(5, ch_out1, ch_out1*6, ch_out, hswish(), SeModule(ch_out), 1),
        )
        
        ch_in, ch_out = int(160*nfts), int(160*nfts) 
        self.block4 = nn.Sequential(
            Block(5, ch_in, ch_in*4, ch_out, hswish(), SeModule(ch_out), 2), # 4.2
            Block(5, ch_out, ch_out*6, ch_out, hswish(), SeModule(ch_out), 1),
        )

#         self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x0,x1,x2,x3,x4

# zfbi
class MobileNetV3_EncoderHighlight(nn.Module):
    def __init__(self, img_ch=1, nfts=1):
        super(MobileNetV3_EncoderHighlight, self).__init__()
        
        self.mode = 'trilinear'
        
        ch_in, ch_out = img_ch, int(16*nfts)
        self.conv0 = nn.Sequential( 
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ch_out),
            hswish(),            
        )
        self.block0 = nn.Sequential(             
            Block(3, ch_out, ch_out*1, ch_out, nn.ReLU(inplace=True), None, 1),
        )        
        
        ch_in, ch_out = int(16*nfts)+int(16*nfts), int(24*nfts)
        self.block1 = nn.Sequential(
            Block(3, ch_in, ch_in*4, ch_out, nn.ReLU(inplace=True), None, 2),
            Block(3, ch_out, ch_out*3, ch_out, nn.ReLU(inplace=True), None, 1),
        )
        
        ch_in, ch_out = int(24*nfts)+int(16*nfts), int(40*nfts)
        self.block2 = nn.Sequential(    
            Block(5, ch_in, ch_in*3, ch_out, nn.ReLU(inplace=True), SeModule(ch_out), 2),
            Block(5, ch_out, ch_out*3, ch_out, nn.ReLU(inplace=True), SeModule(ch_out), 1),
            Block(5, ch_out, ch_out*3, ch_out, nn.ReLU(inplace=True), SeModule(ch_out), 1),
        )
        
        ch_in, ch_out0, ch_out1, ch_out = int(40*nfts)+int(16*nfts), int(80*nfts), int(112*nfts), int(160*nfts)
        self.block3 = nn.Sequential(
            Block(3, ch_in, ch_in*6, ch_out0, hswish(), None, 2),
            Block(3, ch_out0, ch_out0*2, ch_out0, hswish(), None, 1), # 2.5
            Block(3, ch_out0, ch_out0*2, ch_out0, hswish(), None, 1), # 2.3
            Block(3, ch_out0, ch_out0*2, ch_out0, hswish(), None, 1), # 2.3               
            Block(3, ch_out0, ch_out0*6, ch_out1, hswish(), SeModule(ch_out1), 1),
            Block(3, ch_out1, ch_out1*6, ch_out1, hswish(), SeModule(ch_out1), 1),
            Block(5, ch_out1, ch_out1*6, ch_out, hswish(), SeModule(ch_out), 1),
        )
        
        ch_in, ch_out = int(160*nfts)+int(16*nfts), int(160*nfts) 
        self.block4 = nn.Sequential(          
            Block(5, ch_in, ch_in*4, ch_out, hswish(), SeModule(ch_out), 2), # 4.2
            Block(5, ch_out, ch_out*6, ch_out, hswish(), SeModule(ch_out), 1),
        )

#         self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = self.conv0(x)
        x0 = self.block0(x)
        
        x0p = F.interpolate(x,
                            size=tuple(x0.shape[2:]),
                            mode=self.mode,
                            align_corners=True)  
        x1 = self.block1(torch.cat([x0p,x0], dim=1))
        
        x1p = F.interpolate(x, 
                            size=tuple(x1.shape[2:]),
                            mode=self.mode,
                            align_corners=True)    
        
        x2 = self.block2(torch.cat([x1p,x1], dim=1))

        x2p = F.interpolate(x,
                            size=tuple(x2.shape[2:]),
                            mode=self.mode,
                            align_corners=True)        
        x3 = self.block3(torch.cat([x2p,x2], dim=1))                       
                         
        x3p = F.interpolate(x,
                            size=tuple(x3.shape[2:]),
                            mode=self.mode,
                            align_corners=True)        
        x4 = self.block4(torch.cat([x3p,x3], dim=1))
                         
        return x0,x1,x2,x3,x4