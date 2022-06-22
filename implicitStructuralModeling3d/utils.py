import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collections import OrderedDict

from modeling_tools import *
from losses import lossf

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args['optimizer_type'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args['optimizer_type'] == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args['optimizer_type'] == 'Adamax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args['optimizer_type'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args['lr']
    kwargs['weight_decay'] = args['weight_decay']

    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args['decay_type'] == 'StepLR':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    elif args['decay_type'] == 'MultiStepLR':
        milestones = args['decay_type'].split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args['gamma']
        )
    elif args['decay_type'] == 'ReduceLROnPlateau':
        scheduler = lrs.ReduceLROnPlateau(
            my_optimizer, 
            'min', 
            patience=args['lr_decay'], 
            factor=args['gamma'])
    return scheduler

def load_model_parameters(model, checkpoint_path, device):
    OrderedDictOrg = torch.load(checkpoint_path, map_location=device)['state_dict']
    OrderedDictNew = OrderedDict()
    for key in OrderedDictOrg.keys():
        keyNew = '.'.join(key.split('.')[1:])
        OrderedDictNew[keyNew] = OrderedDictOrg[key]
    model.load_state_dict(OrderedDictNew)

def threshold_filter(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def reverse_min_max_norm(x, m, a):
    x = x * (a - m) + m        
    return x

def min_max_norm(x):
    if torch.is_tensor(x) and torch.max(x) != torch.min(x):
            x = x - torch.min(x)
            x = x / torch.max(x)        
    elif np.max(x) != np.min(x):
            x = x - np.min(x)
            x = x / np.max(x)
    return x
    
# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x) and torch.std(x) != 0:
            x = (x - torch.mean(x)) / torch.std(x)
    elif np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
    return x

def create_grad_mask(fl, edge=10, beta=1e-2):
    fl_mk= gaussian_filter(fl, 1.0)
    fl_mk[fl_mk < beta] = 0
    fl_mk = fl_mk.astype(np.bool_)
    eg_mk = np.ones_like(fl_mk, dtype=np.bool_)
    eg_mk[edge:-edge,edge:-edge] = 0
    return ~ (eg_mk | fl_mk)    

# 定义数据集       
class build_dataset_rgt(Dataset):
    def __init__(self, samples_list, dataset_path, mode,
                 possible_num_hrzs, hrz_grp, 
                 bit, bit_mute, bit_rate, 
                 fault_range,
                 use_normal=False, norm=None):
        
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.mode = mode
        
        self.possible_num_hrzs = possible_num_hrzs
        self.hrz_grp = hrz_grp
        self.bit = bit
        self.bit_mute = bit_mute
        self.bit_rate = bit_rate
        self.fault_range = fault_range
        
        self.norm = norm
        self.use_normal = use_normal
    
    def compute_grad_from_numpy(self, rgt, eps=1e-08):     
        grad = compute_grad_from_tensor(torch.tensor(rgt).unsqueeze(0).unsqueeze(0), eps)
        return grad.numpy()[0]
    
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file)
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        ux = sample_dict['rgt']
        if self.norm is not None:
            ux = self.norm(ux)            
        fl = sample_dict['fault']
        
        if self.mode in ['Train', 'Valid']:
            
            if self.use_normal:  
                ox = self.compute_grad_from_numpy(ux)         
                fr, fo, mk = get_train_sample_from_rgt(ux, self.possible_num_hrzs, self.hrz_grp, 
                                                       self.bit, self.bit_mute, self.bit_rate, 
                                                       fl=fl, fl_rg=self.fault_range, 
                                                       orientation=ox, data_sample_path=sample_file_path)
                    
                sample_output['normal'] = fo.astype(np.single)
                sample_output['orientation'] = ox.astype(np.single)
                
            else: 
                fr, mk = get_train_sample_from_rgt(ux, self.possible_num_hrzs, self.hrz_grp,
                                                   self.bit, self.bit_mute, self.bit_rate,
                                                   fl=fl, fl_rg=self.fault_range, 
                                                   data_sample_path=sample_file_path)

        elif self.mode == 'Infer':
                                 
            if self.use_normal:           
                ox = self.compute_grad_from_numpy(ux) 
                
                fr, fo, mk = get_train_sample_from_rgt(ux, self.possible_num_hrzs, self.hrz_grp,
                                                   self.bit, self.bit_mute, self.bit_rate,
                                                   fl=fl, fl_rg=self.fault_range, 
                                                   orientation=ox, data_sample_path=sample_file_path)              
                
                sample_output['normal'] = fo.astype(np.single)
                sample_output['orientation'] = ox.astype(np.single)

            else:
                fr, mk = get_train_sample_from_rgt(ux, self.possible_num_hrzs, self.hrz_grp,
                                                   self.bit, self.bit_mute, self.bit_rate,
                                                   fl=fl, fl_rg=self.fault_range, 
                                                   data_sample_path=sample_file_path)

        sample_output['rgt'] = ux[np.newaxis,:,:].astype(np.single)
        sample_output['scalar'] = fr[np.newaxis,:,:].astype(np.single)
        sample_output['fault'] = fl[np.newaxis,:,:].astype(np.single)

        sample_output["mask_scalar"] = mk[np.newaxis,:,:].astype(np.single)
        sample_output["mask_normal"] = create_grad_mask(fl)[np.newaxis,:,:].astype(np.single)
        
        return  sample_output    

# 读取数据体
def read_cube(data_path, data_file, num_inline, num_crossline):
    data_file = os.path.join(data_path, data_file+".dat")
    print(data_file)
    if os.path.exists(data_file):
        cube = np.load(data_file)
        cube = np.reshape(cube, (num_crossline, num_inline, -1))
        return cube
    else:
        return None

# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def compute_grad_from_tensor(data, eps=1e-8):

    if data.shape[0] > 1:
        grad_i = torch.stack(torch.gradient(data.tile(1,2,1,1), dim=[2,3])).permute(2,1,0,3,4)[0]
    else:
        grad_i = torch.stack(torch.gradient(data.tile(2,2,1,1), dim=[2,3])).permute(2,1,0,3,4)[0,:1]

    grad_k = torch.sqrt(grad_i.pow(2).sum(dim=1)) + eps
    return grad_i / grad_k.unsqueeze(1)

# 训练和验证
def train_valid_net(param, model, train_data, valid_data, plot=True, use_normal=False, device=None):
    
    #初始化参数
    inshape = param['inshape']
    epochs = param['epochs']
    batch_size = param['batch_size']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, drop_last=True)
    
    optimizer = make_optimizer(param, model) 
    scheduler = make_scheduler(param, optimizer)
    
    loss_func_type, loss_func_coef = dict(), param['loss_type']
    for key in param['loss_type'].keys():
        if key == "ms-ssim":
            loss_func_type[key] = lossf.MSSIMLoss(channel=1, filter_size=5).to(device)   
        if key == "ssim":
            loss_func_type[key] = lossf.NSSIMLoss(channel=1, filter_size=5).to(device)              
        elif key in ["mae", "l1"]:
            loss_func_type["mae"] = lossf.MAELoss(reduction='mean').to(device)
            loss_func_coef["mae"] = loss_func_coef[key]
        elif key in ["mse", "l2"]:
            loss_func_type["mse"] = lossf.MSELoss(reduction='mean').to(device)
            loss_func_coef["mse"] = loss_func_coef[key]
        elif key == "csim":
            loss_func_type[key] = lossf.CSLoss(inshape, reduction='sum', dim=1).to(device)      
    
    # 主循环
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    best_mse = 1e50
    warm_epoch = 2

    logs = []
    for epoch in range(epochs):
        
        # 训练阶段
        model.train()
        
        loss_train_per_epoch = dict() 
        for key in loss_func_type.keys():
            loss_train_per_epoch[key] = 0.0
        loss_train_per_epoch["loss"] = 0.0
        
        for batch_idx, batch_samples in enumerate(train_loader):            
            
            # fault
            fault = batch_samples["fault"]
            fault = Variable(fault.to(device))
            
            # rgt
            rgt = batch_samples["rgt"]
            rgt = Variable(rgt.to(device))
            
            # scarlar
            mask_scalar = batch_samples["mask_scalar"]
            mask_scalar = Variable(mask_scalar.to(device))
            
            scalar = rgt * mask_scalar
            
            optimizer.zero_grad()
            
            # orientation
            if use_normal:
                orientation = batch_samples["orientation"]
                orientation = Variable(orientation.to(device))

                mask_normal = batch_samples["mask_normal"]
                mask_normal = Variable(mask_normal.to(device))
                
                normal = orientation * mask_scalar  
                
                data = torch.cat((scalar, normal, fault), dim=1)
                rgt_i = model(data)
                orientation_i =  compute_grad_from_tensor(rgt_i)
                orientation_j =  orientation_i * (1 - mask_scalar) + orientation * mask_scalar                  
                
            else:
                data = torch.cat((scalar, fault), dim=1)
                rgt_i = model(data)

            rgt_j =  rgt_i * (1 - mask_scalar) + rgt * mask_scalar      
              
            for i, key in enumerate(loss_func_type.keys()):
                if key == "csim":
                    if epoch <= warm_epoch:
                        continue
                    batch_loss = loss_func_type[key](orientation_j, orientation, mask_normal)
                elif key in ["mae", "mse"]:
                    batch_loss = loss_func_type[key](rgt_i, rgt)
                else:
                    batch_loss = loss_func_type[key](rgt_j, rgt)
                    
                if i == 0 :
                    loss  = batch_loss * loss_func_coef[key]
                else:
                    loss += batch_loss * loss_func_coef[key] 
                loss_train_per_epoch[key] += batch_loss.item()            
                
            loss.backward()  
            optimizer.step()
            
            loss_train_per_epoch['loss'] += loss.item()
            
        # 验证阶段
        model.eval()
        
        loss_valid_per_epoch = dict() 
        for key in loss_func_type.keys():
            loss_valid_per_epoch[key] = 0.0
        loss_valid_per_epoch["loss"] = 0.0  
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):   

                # fault
                fault = batch_samples["fault"]
                fault = Variable(fault.to(device))

                # rgt
                rgt = batch_samples["rgt"]
                rgt = Variable(rgt.to(device))

                # scarlar
                mask_scalar = batch_samples["mask_scalar"]
                mask_scalar = Variable(mask_scalar.to(device))
            
                scalar = rgt * mask_scalar
                
                # orientation
                if use_normal:
                    orientation = batch_samples["orientation"]
                    orientation = Variable(orientation.to(device))
                    mask_normal = batch_samples["mask_normal"]
                    mask_normal = Variable(mask_normal.to(device))

                    normal = orientation * mask_scalar  
                    data = torch.cat((scalar, normal, fault), dim=1)
                    rgt_i = model(data)
                    orientation_i =  compute_grad_from_tensor(rgt_i)
                    orientation_j =  orientation_i * (1 - mask_scalar) + orientation * mask_scalar  
                    
                else:
                    data = torch.cat((scalar, fault), dim=1)                
                    rgt_i = model(data)
       
                rgt_j =  rgt_i * (1 - mask_scalar) + rgt * mask_scalar                
                    
                for i, key in enumerate(loss_func_type.keys()):
                    if key == "csim":
                        if epoch <= warm_epoch:
                            continue
                        batch_loss = loss_func_type[key](orientation_j, orientation, mask_normal)
                    elif key in ["mae", "mse"]:
                        batch_loss = loss_func_type[key](rgt_i, rgt)
                    else:
                        batch_loss = loss_func_type[key](rgt_j, rgt)
                        
                    if i == 0 :
                        loss  = batch_loss * loss_func_coef[key]
                    else:
                        loss += batch_loss * loss_func_coef[key] 
                        
                    loss_valid_per_epoch[key] += batch_loss.item()  
                loss_valid_per_epoch["loss"] += loss.item()  
        
        log_per_epoch = {"learning_rate":optimizer.param_groups[0]['lr'],
                          "epoch":epoch}
        for key in loss_train_per_epoch.keys():      
            log_per_epoch[key+"_train"] = loss_train_per_epoch[key] / len(train_loader)
            log_per_epoch[key+"_valid"] = loss_valid_per_epoch[key] / len(valid_loader) 
        logs.append(log_per_epoch)
        
        epoch_loss_train.append(log_per_epoch["loss_train"])
        epoch_loss_valid.append(log_per_epoch["loss_train"])
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        
        # 保存训练日志
        if epoch % save_inter == 0:
            np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
        
        # 保存中间模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            
        # 保存最优模型
        if epoch_loss_valid[epoch] < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = epoch_loss_valid[epoch]

        scheduler.step(log_per_epoch["loss_train"])

        # 显示loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.8f} Validation Loss:{:.8f} Learning rate: {:.8f}'.format(epoch, epoch_loss_train[epoch], epoch_loss_valid[epoch], epoch_lr[epoch])) 
            
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    return model

def pred(model, test_data, use_normal=False, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    model.eval()
    output_pred_samples = [] 
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(test_loader):

            # fault
            fault = batch_samples["fault"]
            fault = Variable(fault.to(device))

            # rgt
            rgt = batch_samples["rgt"]
            rgt = Variable(rgt.to(device))
            
            # scarlar
            mask_scalar = batch_samples["mask_scalar"]
            mask_scalar = Variable(mask_scalar.to(device))            
            
            scalar = rgt * mask_scalar
            
            # orientation
            if use_normal:
                orientation = batch_samples["orientation"]
                orientation = Variable(orientation.to(device))

                mask_normal = batch_samples["mask_normal"]
                mask_normal = Variable(mask_normal.to(device))
                
                normal = orientation * mask_scalar  
                
                data = torch.cat((scalar, normal, fault), dim=1)
            else:
                data = torch.cat((scalar, fault), dim=1)       
                
            rgt_i = model(data)
            
            pred_sample = {} 
            if use_normal:
                orientation_i = compute_grad_from_tensor(rgt_i)
                pred_sample["normal"] = normal.cpu().squeeze(0).numpy() 
                pred_sample["mask_normal"] = batch_samples["mask_normal"].squeeze(0).numpy()                   
                pred_sample["orientation"] = compute_grad_from_tensor(rgt).cpu().squeeze(0).numpy()  
                pred_sample["orientation_pred"] = orientation_i.cpu().squeeze(0).numpy()
            
            pred_sample["pred"] = rgt_i.cpu().squeeze(0).numpy()   
            pred_sample["rgt"] = batch_samples["rgt"].squeeze(0).numpy()    
            pred_sample["scalar"] = scalar.cpu().squeeze(0).numpy()    
            pred_sample["fault"] = batch_samples["fault"].squeeze(0).numpy()   
            pred_sample["mask_scalar"] = batch_samples["mask_scalar"].squeeze(0).numpy()
            
            output_pred_samples.append(pred_sample)
            
    return output_pred_samples
