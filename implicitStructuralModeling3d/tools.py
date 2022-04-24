import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf

from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage.filters import gaussian_filter

def write_ps_data(ps_data_path, nps, fps):

    if not os.path.exists(ps_data_path):
        os.makedirs(ps_data_path)

    fps0 = fps[0]
    fault_ps_path = os.path.join(ps_data_path, "fault.txt")
    with open(fault_ps_path, 'w+') as f:
        f.write(f"X Y Z PART_ID\n")
        for i in range(len(fps0['x'])):
            f.write(f"{fps0['x'][i]} {fps0['y'][i]} {fps0['z'][i]} {1}\n")

    for i,_nps in enumerate(nps):
        horizon_ps_path = os.path.join(ps_data_path, f"horizon_{i}.txt")        
        with open(horizon_ps_path, 'w+') as f:
            f.write(f"X Y Z PART_ID\n")
            for j in range(len(_nps['x'])):
                f.write(f"{_nps['x'][j]} {_nps['y'][j]} {_nps['z'][j]} {i}\n") 

def cut_cube(x,a,b):         
    return x[a[0]:a[0]+b[0],
            a[1]:a[1]+b[1],
            a[2]:a[2]+b[2]]

def get_values_not_on_grid(image, point_set, shape, mode='bilinear'):
    
    new_locs = point_set.type(torch.FloatTensor)
    if len(shape) > 1:
        new_locs = new_locs.unsqueeze(-1)
    if len(shape) > 2:
        new_locs = new_locs.unsqueeze(-1)
    
    # need to normalize grid values to [-1, 1] for resampler
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    # move channels dim to last position
    # also not sure why, but the channels need to be reversed
    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]        

    values = nnf.grid_sample(image, new_locs,
                          align_corners=True, mode=mode)
    
    return values.squeeze(-1).squeeze(-1)

def thin_fault(fl, size = (6, 6)):
    import cv2
    return cv2.erode(cv2.dilate(fl, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)), 
                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(map(lambda x:x-1, size))))

def min_max_norm(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x 

def mask_horizons_func(z, y, x, hrz_grp=4, hrz_sel=None, sample_rate=1):

    if hrz_sel is None:
        hrz_sel = random.randint(hrz_grp//2, hrz_grp)

    hrz_t, hrz_b = np.min(z),np.max(z) 
    hrz_v = (hrz_b - hrz_t) / hrz_grp
    
    hrz_ids = random.sample(list(range(hrz_grp)), hrz_sel)
                        
    new_z = np.array([], dtype=np.int)
    new_y = np.array([], dtype=np.int)
    new_x = np.array([], dtype=np.int)                                 
    for i in hrz_ids:              
        new_ids = np.where((z>=hrz_t+i*hrz_v) & (z < hrz_t+(i+1)*hrz_v))[0]  
                        
        new_ids = random.sample(new_ids.tolist(), len(new_ids)//sample_rate)
        new_ids = np.array(new_ids, dtype=np.int)   
                        
        new_z = np.append(new_z, z[new_ids])
        new_y = np.append(new_y, y[new_ids])
        new_x = np.append(new_x, x[new_ids])
        
    return new_z, new_y, new_x   
                           
                        
def get_train_sample(rgt, num_hrzs, bit=256, bit_rate=2, bit_mute=60, 
                     fl=None, fl_rg=1, 
                     mask_grp_sel=None, 
                     sample_rate=1,
                     ox=None):
    
    fl_pad = np.pad(fl,((fl_rg,fl_rg),(fl_rg,fl_rg),(fl_rg,fl_rg)), mode='reflect')

    num_valid_hrzs = bit - 1
    
    ux = min_max_norm(rgt) * num_valid_hrzs
    
    valid_hrzs_index = np.sort(np.unique(np.round(ux)))
    mute_min = np.where(valid_hrzs_index <= bit_mute)[0].max()
    mute_max = np.where(valid_hrzs_index > bit-bit_mute)[0].min() 
    valid_hrzs_index = valid_hrzs_index[mute_min:mute_max].tolist()                        

    valid_hrzs_index = [d for i,d in enumerate(valid_hrzs_index) if i%bit_rate == 0] 
    
    hrzs_iterv = len(valid_hrzs_index) // num_hrzs
    
    fx = np.zeros(ux.shape)  
    fm = np.zeros(ux.shape)
    fo = np.zeros((3, *ux.shape)) if ox is not None else None
    ps = list()
    
    max_hrzs_itert = 10                      
                        
    for k in range(num_hrzs):
        
        itert = 0
        num_hrzs_sample = 0
        while num_hrzs_sample == 0:     
                        
            hrzs_index = random.choice(valid_hrzs_index[k*hrzs_iterv:(k+1)*hrzs_iterv])
            z, y, x = np.where((ux >= hrzs_index - bit_rate/2) & (ux < (hrzs_index + bit_rate/2)))   
            
            # remove points near faults 
            new_z, new_y, new_x = [],[],[]
            for j in list(range(len(z))):          
                if fl_pad[z[j]:z[j]+fl_rg*2, 
                      y[j]:y[j]+fl_rg*2,
                      x[j]:x[j]+fl_rg*2].any() > 0:
                    continue    
                new_z.append(z[j])
                new_y.append(y[j])
                new_x.append(x[j])             
            new_z = np.array(new_z, dtype=np.int)
            new_y = np.array(new_y, dtype=np.int) 
            new_x = np.array(new_x, dtype=np.int)

            # mask partial points again
            if mask_grp_sel is not None:
                new_z, new_y, new_x = mask_horizons_func(new_z, new_y, new_x, 
                                                         hrz_grp=mask_grp_sel[0], 
                                                         hrz_sel=mask_grp_sel[1],
                                                         sample_rate = sample_rate)      
            
            num_hrzs_sample = len(new_z)                          
            if itert > max_hrzs_itert:
               break
            else:
               itert += 1            

        fx[new_z, new_y, new_x] = rgt[new_z, new_y, new_x]
        if fo is not None:               
            fo[:, new_z, new_y, new_x] = ox[:, new_z, new_y, new_x]     
        fm[new_z, new_y, new_x] = 1      
                                     
        ps.append({"x":new_x, 
                   "y":new_y, 
                   "z":new_z, 
                  })   
    
    return fx, fo, fm, ps

def get_hrzs_from_volume(ux, hv):
    if isinstance(hv, list):
        hv = np.array(hv)
    n1,n2,n3 = ux.shape
    hrzs = np.zeros((len(hv),n2,n3))
    for i3 in range(n3):
        for i2 in range(n2):
            c0 = ux[:,i2,i3]
            for i,v in enumerate(hv):
                if v <= np.max(c0) and v >= np.min(c0):
                    c0, x0 = np.unique(c0, return_index=True)
                    i1 = np.argsort(c0, axis=0)
                    x0 = np.take_along_axis(x0, i1, axis=0)
                    c0 = np.take_along_axis(c0, i1, axis=0)
                    c0 = gaussian_filter(c0, sigma=4)
                    x0 = gaussian_filter(x0, sigma=4)
                    f = interp1d(c0, x0, fill_value=0, bounds_error=False)
                    hrzs[i,i2,i3] = f(v)                  
    return hrzs
