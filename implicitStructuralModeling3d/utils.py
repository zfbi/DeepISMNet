import os
import math
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collections import OrderedDict

import tools
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

def extract_cube(x,a,b):  
    return x[a[0]:a[0]+b[0],
            a[1]:a[1]+b[1],
            a[2]:a[2]+b[2]]

def write_cube(data, path):                                   
    data = np.transpose(data,[2,1,0]).astype(np.single)
    data.tofile(path)

def interpData2d(a, size, kind="linear"):
    h, w = size
    h0, w0 = a.shape
    y0 = np.linspace(0, h0-1, h0)
    x0 = np.linspace(0, w0-1, w0)
    f = interp2d(x0, y0, a, kind=kind)
    y = np.linspace(0, h0-1, h)
    x = np.linspace(0, w0-1, w)
    return f(x, y)

def interpData1d(a, h, kind="linear"):
    h0 = a.shape[0]
    x0 = np.linspace(0, h0-1, h0)
    f = interp1d(x0, a, kind=kind)
    x = np.linspace(0, h-1, h)
    return f(x)

def read_data(size,path,mode="<f",order='F'):
    data = np.fromfile(path, dtype=mode)
    data = data.reshape(size, order=order)        
    return data                                     

def threshold_filter(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def min_max_norm(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x
    
def mea_std_norm(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
    
def compute_grad_from_tensor(data, eps=1e-8):

    if data.shape[0] > 1:
        grad_i = torch.stack(torch.gradient(data.tile(1,2,1,1,1), dim=[2,3,4])).permute(2,1,0,3,4,5)[0]
    else:
        grad_i = torch.stack(torch.gradient(data.tile(2,2,1,1,1), dim=[2,3,4])).permute(2,1,0,3,4,5)[0,:1]

    grad_k = torch.sqrt(grad_i.pow(2).sum(dim=1)) + eps
    return grad_i / grad_k.unsqueeze(1)
  
def create_grad_mask(fl, edge=10, beta=1e-2):
    fl_mk= gaussian_filter(fl, 1.0)
    fl_mk[fl_mk < beta] = 0
    fl_mk = fl_mk.astype(np.bool_)
    eg_mk = np.ones_like(fl_mk, dtype=np.bool_)
    eg_mk[edge:-edge,edge:-edge,edge:-edge] = 0
    return ~ (eg_mk | fl_mk)    
       
class build_dataset(Dataset):
    def __init__(self, 
                 inshape,
                 samples_list, 
                 dataset_path,
                 mode,
                 num_hrzs_list,
                 bit,
                 mask_grp_sel,
                 bit_rate,
                 sample_rate_list,
                 fault_range,
                 bit_mute,
                 norm=None,
                 point_set=False,
                 use_normal=False):
        
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.norm = norm
        self.mode = mode
        self.inshape = inshape
        
        self.num_hrzs_list = num_hrzs_list
        self.bit = bit
        self.mask_grp_sel = mask_grp_sel
        self.bit_rate = bit_rate
        self.sample_rate_list = sample_rate_list
        self.fault_range = fault_range
        self.bit_mute = bit_mute
        
        self.point_set = point_set
        self.use_normal = use_normal
        
    def __len__(self):
        return len(self.samples_list)

    def compute_grad_from_numpy(self, rgt, eps=1e-08):     
        grad = compute_grad_from_tensor(torch.tensor(rgt).unsqueeze(0).unsqueeze(0), eps)
        return grad.numpy()[0]    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file)
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        num_hrzs = random.choice(self.num_hrzs_list)
        sample_rate = random.choice(self.sample_rate_list)
        
        ux = sample_dict['rgt']
        fl = sample_dict['fault']

        starts = list(map(lambda x,y:random.randint(0,x-y), ux.shape, self.inshape))
        
        fl = extract_cube(fl,starts,self.inshape)
        ux = extract_cube(ux,starts,self.inshape)
        
        ux = self.norm(ux) if self.norm is not None else ux
        
        if self.mode in ['Train', 'Valid']:
  
            if self.use_normal is not None:
                ox = self.compute_grad_from_numpy(ux)      
            else:
                ox=None
                
            _, _, fm, ps = tools.get_train_sample(ux, num_hrzs, bit=self.bit, 
                                    bit_rate=self.bit_rate, bit_mute=self.bit_mute,  
                                    fl=fl, fl_rg=self.fault_range,
                                    mask_grp_sel=self.mask_grp_sel, sample_rate=sample_rate,
                                    ox=ox)                 

        elif self.mode == 'Infer':
               
            if self.use_normal is not None:
                ox = self.compute_grad_from_numpy(ux)      
            else:
                ox=None
                
            _, _, fm, ps = tools.get_train_sample(ux, num_hrzs, bit=self.bit, 
                                    bit_rate=self.bit_rate, bit_mute=self.bit_mute,  
                                    fl=fl, fl_rg=self.fault_range,
                                    mask_grp_sel=self.mask_grp_sel, sample_rate=sample_rate,
                                    ox=ox)          
        
        sample_data = {}    
        sample_data['fault'] = fl[np.newaxis,:,:,:].astype(np.single)
        sample_data['rgt'] = ux[np.newaxis,:,:,:].astype(np.single)
        
        sample_data["mask"]  = fm[np.newaxis,:,:,:].astype(np.single)
        
        if self.use_normal is not None: 
            sample_data['orientation'] = ox[:,:,:,:].astype(np.single) 
            
            mk = create_grad_mask(fl) # & (~ fm.astype(np.bool_))
            sample_data['mask_normal'] = mk[np.newaxis,:,:,:].astype(np.single) 
        
        if self.point_set:
            sample_data['point_set_scalar'] = ps
            
            z, y, x  = np.where(fl>0)
            sample_data['point_set_fault'] = [{"x":x,"y":y,"z":z}]
            
            if self.use_normal is not None: 
                orn_keys = ["x","y","z","u","v","w"]
                ps_normal = dict()
                for key in orn_keys:
                    ps_normal[key] = np.array([])
                for point_set in ps:
                    for key in orn_keys[:3]:
                        ps_normal[key] = np.append(ps_normal[key], point_set[key]) 
                    for k, key in enumerate(orn_keys[3:]):
                        ps_normal[key] = np.append(ps_normal[key], 
                                                   ox[k,
                                                      ps_normal["z"].astype(np.int32),
                                                      ps_normal["y"].astype(np.int32),
                                                      ps_normal["x"].astype(np.int32)])                         
                sample_data['point_set_normal'] = [ps_normal]
        
        return  sample_data    

def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train_valid_net(param, model, train_data, valid_data, plot=True, device=None, use_normal=False, warm_epoch=2):

    epochs = param['epochs']
    batch_size = param['batch_size']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    
    optimizer = make_optimizer(param, model) 
    scheduler = make_scheduler(param, optimizer)
    
    loss_func_type, loss_func_coef = dict(), param['loss_type']
    for key in param['loss_type'].keys():
        if key == "ms-ssim":
            loss_func_type["ms-ssim"] = lossf.MSSIMLoss(channel=1, filter_size=5).to(device)
        elif key in ["mae", "l1"]:
            loss_func_type["mae"] = lossf.MAELoss(reduction='mean').to(device)
            loss_func_coef["mae"] = loss_func_coef[key]
        elif key in ["mse", "l2"]:
            loss_func_type["mse"] = lossf.MSELoss(reduction='mean').to(device)
            loss_func_coef["mse"] = loss_func_coef[key]
        elif key == "csim":
            loss_func_type["cosine"] = lossf.CSLoss(inshape, reduction='sum', dim=1).to(device)            

    logs = list()
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], [] # plot
    best_mse = 1e50
    
    for epoch in range(epochs):

        model.train()
        
        loss_train_per_epoch = dict() 
        for key in loss_func_type.keys():
            loss_train_per_epoch[key] = 0.0
        loss_train_per_epoch["loss"] = 0.0
        
        for batch_idx, batch_samples in enumerate(train_loader):

            fault = batch_samples["fault"]
            fault = Variable(fault.to(device))
            
            rgt = batch_samples["rgt"]
            rgt = Variable(rgt.to(device))

            mask = batch_samples["mask"]
            mask = Variable(mask.to(device))            

            scalar = rgt * mask  
            
            optimizer.zero_grad()
            
            # orientation
            if use_normal:
                orientation = batch_samples["orientation"]
                orientation = Variable(orientation.to(device))
                normal = orientation * mask  
                mask_normal = batch_samples["mask_normal"]
                mask_normal = Variable(mask_normal.to(device))
                
                data = torch.cat([scalar, normal, fault], dim=1)
                rgt_i = model(data)
                orientation_j = compute_grad_from_tensor(rgt_j)
                orientation_j = orientation_i * (1 - mask) + orientation * mask
                
            else:
                data = torch.cat([scalar, fault], dim=1)
                
                rgt_i = model(data)
                
            rgt_j =  rgt_i * (1 - mask) + rgt * mask
                
            for i, key in enumerate(loss_func_type.keys()):
                if key == "csim" and epoch <= warm_epoch:
                    continue
                batch_loss = loss_func_type[key](rgt_i, rgt)   
                if i == 0 :
                    loss  = batch_loss * loss_func_coef[key]
                else:
                    loss += batch_loss * loss_func_coef[key]  
                loss_train_per_epoch[key] += batch_loss.item()
                                
            loss.backward()
            optimizer.step()
            
            loss_train_per_epoch['loss'] += loss.item()
            
        model.eval()

        loss_valid_per_epoch = dict() 
        for key in loss_func_type.keys():
            loss_valid_per_epoch[key] = 0.0
        loss_valid_per_epoch["loss"] = 0.0        
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):   

                fault = batch_samples["fault"]
                fault = Variable(fault.to(device))

                rgt = batch_samples["rgt"]
                rgt = Variable(rgt.to(device))

                mask = batch_samples["mask"]
                mask = Variable(mask.to(device)) 
                
                scalar = rgt * mask    
            
                # orientation
                if use_normal:
                    orientation = batch_samples["orientation"]
                    orientation = Variable(orientation.to(device))
                    normal = orientation * mask  
                    mask_normal = batch_samples["mask_normal"]
                    mask_normal = Variable(mask_normal.to(device))
                    
                    data = torch.cat([scalar, normal, fault], dim=1)
                    
                    rgt_i = model(data)

                    orientation_j = compute_grad_from_tensor(rgt_j)  
                    orientation_j = orientation_i * (1 - mask) + orientation * mask
                    
                else:
                    data = torch.cat([scalar, fault], dim=1)
                    rgt_i = model(data)
                rgt_j =  rgt_i * (1 - mask) + rgt * mask
           
            for i, key in enumerate(loss_func_type.keys()):
                if key == "csim" and epoch <= warm_epoch:
                    continue                 
                batch_loss = loss_func_type[key](rgt_i, rgt)
                if i == 0 :
                    loss  = batch_loss * loss_func_coef[key]
                else:
                    loss += batch_loss * loss_func_coef[key] 
                loss_valid_per_epoch[key] += batch_loss.item()
            loss_valid_per_epoch['loss'] += loss.item()
         
        log_per_epoch = {"learning_rate":optimizer.param_groups[0]['lr'],
                          "epoch":epoch}
        for key in loss_train_per_epoch.keys():      
            log_per_epoch[key+"_train"] = loss_train_per_epoch[key] / len(train_loader)
            log_per_epoch[key+"_valid"] = loss_valid_per_epoch[key] / len(valid_loader) 
        logs.append(log_per_epoch)
        
        epoch_loss_train.append(log_per_epoch["loss_train"])
        epoch_loss_valid.append(log_per_epoch["loss_valid"])
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
            
        if epoch_loss_valid[epoch] < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = epoch_loss_valid[epoch]

        scheduler.step(log_per_epoch["loss_train"])

        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.8f} Validation Loss:{:.8f} Learning rate: {:.8f}'.format(epoch, epoch_loss_train[epoch], epoch_loss_valid[epoch], epoch_lr[epoch]))           

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
    
    def point_set_filter(ps):
        for i in range(len(ps)):
            for key in ps[i].keys():
                ps[i][key] = ps[i][key].numpy()[0]
        return ps    
    
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
            
            # mask
            mask = batch_samples["mask"]
            mask = Variable(mask.to(device))            
            
            scalar = rgt * mask    

            # orientation
            if use_normal:
                orientation = batch_samples["orientation"]
                orientation = Variable(orientation.to(device))
                normal = orientation * mask  
                mask_normal = batch_samples["mask_normal"]
                mask_normal = Variable(mask_normal.to(device))
                data = torch.cat([scalar, normal, fault], dim=1)
                rgt_i = model(data)
                orientation_i = utils.compute_grad_from_tensor(rgt_i) 
 
            else:
                data = torch.cat([scalar, fault], dim=1)
                rgt_i = model(data)
                
            pred_sample = {}
            pred_sample["pred"] = rgt_i.detach().cpu().squeeze(0).numpy()      
            pred_sample["rgt"] = batch_samples["rgt"].squeeze(0).numpy()      
            pred_sample["fault"] = batch_samples["fault"].squeeze(0).numpy()   
            pred_sample["mask"] = batch_samples["mask"].squeeze(0).numpy()
            pred_sample["point_set_scalar"] = point_set_filter(batch_samples["point_set_scalar"])
            pred_sample["point_set_fault"] = point_set_filter(batch_samples["point_set_fault"])            
            if use_normal:
                pred_sample["orientation"] = batch_samples["orientation"].squeeze(0).numpy()
                pred_sample["mask_normal"] = batch_samples["mask_normal"].squeeze(0).numpy()
                pred_sample["pred_orientation"] = orientation_i.detach().cpu().squeeze(0).numpy() 
                pred_sample["point_set_normal"] = point_set_filter(batch_samples["point_set_normal"])
            output_pred_samples.append(pred_sample)
            
    return output_pred_samples