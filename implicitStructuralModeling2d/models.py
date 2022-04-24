import torch
import torch.nn.functional as F
import torch.nn as nn

import unets 
    
class AttUNet(nn.Module):
    def __init__(self, param):
        super(AttUNet, self).__init__()
        self.encoder_decoder = unets.AttU_Net(img_ch=param['input_channels'], 
                                          output_ch=param['output_channels'])
    def forward(self, x):
        x = self.encoder_decoder(x)
        return x      

class UNet(nn.Module):
    def __init__(self, param):
        super(UNet, self).__init__()
        self.encoder_decoder = unets.U_Net(img_ch=param['input_channels'], 
                                          output_ch=param['output_channels'])
    def forward(self, x):
        x = self.encoder_decoder(x)
        return x    


class ISMNet(nn.Module):
    def __init__(self, param):
        super(ISMNet, self).__init__()
        self.encoder_decoder = unets.ISMNet(param['input_channels'], param['output_channels'])
    def forward(self, x):
        x = self.encoder_decoder(x)
        return x      