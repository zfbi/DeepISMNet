import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from losses.ssim import MultiScaleSSIMLoss, SSIMLoss

class CSLoss(nn.Module):
    def __init__(self, inshape, reduction='mean', dim=1):
        super(CSLoss, self).__init__()
        
        self.mse = nn.MSELoss(reduction=reduction)
        self.cosine_similarity = nn.CosineSimilarity(dim=dim)

        normal = torch.zeros([2, *inshape])
        normal[0] = 1
        normal = normal.type(torch.FloatTensor)  
        self.register_buffer('normal', normal)
        self.reduction = reduction
        
    def forward(self, output, target, mask=None):       
        normal = self.normal.tile([output.size(0),1,1,1]) 
        cosine_output = 1 - self.cosine_similarity(output, normal).unsqueeze(1)
        cosine_target = 1 - self.cosine_similarity(target, normal).unsqueeze(1)
        if mask is not None and self.reduction == "sum":
            loss = self.mse(output * mask, target * mask) / mask.sum()
        else:
            loss = self.mse(output, target)   

        return loss
        
class MSSIMLoss(nn.Module):
    def __init__(self, channel, filter_size):
        super(MSSIMLoss, self).__init__()
        self.mssim = MultiScaleSSIMLoss(channel=channel, filter_size=filter_size)
    def forward(self, output, target):
        loss = (1 - self.mssim(output, target))
        return loss

class NSSIMLoss(nn.Module):
    def __init__(self, channel, filter_size):
        super(NSSIMLoss, self).__init__()
        self.ssim = NSSIMLoss(channel=channel, filter_size=filter_size)
    def forward(self, output, target):
        loss = (1 - self.ssim(output, target))
        return loss    
    
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    def forward(self, output, target, mask=None):
        loss = self.mse(output*mask, target*mask) / mask.sum() if mask is not None else self.mse(output, target)
        return loss      

class MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss(reduction=reduction)
    def forward(self, output, target, mask=None):
        loss = self.mae(output*mask, target*mask) / mask.sum() if mask is not None else self.mae(output, target)
        return loss     

class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.mae = nn.SmoothL1Loss(reduction=reduction, beta=beta)
    def forward(self, output, target, mask=None):
        loss = self.mae(output*mask, target*mask) / mask.sum() if mask is not None else self.mae(output, target)
        return loss    

class Grad(nn.Module):

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
