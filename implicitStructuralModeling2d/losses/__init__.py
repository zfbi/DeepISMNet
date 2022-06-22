import torch
import torch.nn as nn
from importlib import import_module

from . import lossf as loss_func

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, device):
        super(Loss, self).__init__()

        self.loss = []
        self.regularize = []
        self.loss_module = nn.ModuleList()

        for loss in args['loss'].split('+'):
                
            weight, loss_type = loss.split('*')
            
            if loss_type.upper() == 'MSE':
                loss_function = loss_func.MSELoss()
            elif loss_type.upper() == 'MSSIM':
                loss_function = loss_func.MSSIMLoss(1, 3) 
            elif loss_type.upper() == 'SSIM':
                loss_function = loss_func.SSIMLoss(1, 3)                 
            elif loss_type.find('GAN') >= 0:
                loss_function = loss_func.AdversarialLoss(
                    args,
                    loss_type
                )
        
            elif loss_type.upper() in ['L1', 'L2']:
                loss_function = loss_func.Grad(loss_type, loss_mult=args['int_downsize'])
                self.regularize.append({
                    'type': loss_type,
                    'weight': float(weight),
                    'function':loss_function}
                )
                continue
 
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function})
                
        print("Loss function definition:")
        for lf in self.loss:
            if lf['function'] is not None:
                print('\t{:.3f} * {}'.format(lf['weight'],
                                             lf['type']))         
                self.loss_module.append(lf['function'])               
                
        for rf in self.regularize:
            if rf['function'] is not None:
                print('\t{:.3f} * {}'.format(rf['weight'], 
                                             rf['type']))
                self.loss_module.append(rf['function'])
            
        self.loss_module.to(device)

    def forward(self, oup, gt):
        losses = []
        log = {}
        for lf in self.loss:           
            if lf['function'] is not None:
                    
                loss = lf['function'](oup, gt)  
                    
                effective_loss = lf['weight'] * loss
                losses.append(effective_loss)                  
                log[lf['type']] = loss.item()
                
        for rf in self.regularize:
            if rf['function'] is not None: 
                
                loss = rf['function'](oup)
                effective_loss = rf['weight'] * loss
                losses.append(effective_loss) 
                log[rf['type']] = loss.item()
                
        loss_sum = sum(losses)
        return loss_sum, log
