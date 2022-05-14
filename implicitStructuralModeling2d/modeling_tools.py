import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

def sampling_density(img, sampling_rate):
    x, y = np.where(img > 0)
    sampleing_num = round(len(x) * sampling_rate)
    d = np.random.choice(len(x), sampleing_num, replace=False)
    img_new = np.zeros(img.shape)
    for i in d:
        img_new[x[i],y[i]] = img[x[i],y[i]] 
    return img_new

def min_max_norm(x):     
    x = x - np.min(x)
    x = x / np.max(x)
    return x

def compute_hrzs_not_on_grid(fx, hms):          
    hrzs = []
    for hm in hms:
        fm = fx * hm
        i1s,i2s = [],[]
        n1,n2 = fm.shape
        for i2 in range(n2):
            tmp = np.where(fm[:,i2]>0)[0]
            if len(tmp):
                i1s.append(tmp.mean())
                i2s.append(i2)
        hrzs.append([i1s,i2s])
    return cut_lins_for_each_hrz(hrzs)

def cut_lins_for_each_hrz(lines):
    hrzs = []
    for line in lines:
        hrz = []
        if len(line[0]) <= 1:
            continue
        line = np.array(line)
        idxs = np.argsort(line[1])
        i1s = line[0][idxs]
        i2s = line[1][idxs]
        ib = 0
        for i in range(len(i2s)-1):
            if i2s[i+1]>i2s[i]+4:
                hrz.append([i1s[ib:i+1],i2s[ib:i+1]])
                ib = i+1               
        hrz.append([i1s[ib:i+1],i2s[ib:i+1]])
        hrzs.append(hrz)
    return hrzs

def compute_edist(p0, p1):
    return ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5

def map_hrzs_into_img(hrzs, size, rg=0.6):
    h, w = size
    img = np.zeros(size)
    hrzv = []
    for hrz in hrzs:
        v, c = 0, 0
        if isinstance(hrz,tuple):
            hrz = [hrz]
        for i1s,_ in hrz:
            v += i1s.sum()
            c += len(i1s)
        hrzv.append(v / c)

    for i1 in range(h):
        for i2 in range(w):      
            for i,hrz in enumerate(hrzs):
                if isinstance(hrz,tuple):
                    hrz = [hrz]                
                for j1s,j2s in hrz:
                    for j1,j2 in zip(j1s,j2s):
                        dist = compute_edist((i1,i2),(j1,j2))
                        if dist <= rg:
                            if i1 > j1:
                                img[i1][i2] = hrzv[i] + dist
                            else:
                                img[i1][i2] = hrzv[i] - dist
    return img/(h-1)    

def map_fals_into_img(faults, size, rg=1.2):
    h, w = size
    img = np.zeros(size)
    for i1 in range(h):
        for i2 in range(w):      
            for i,fault in enumerate(faults):
                j1s,j2s = fault 
                for j1,j2 in zip(j1s,j2s): 
                    dist = compute_edist((i1,i2),(j1,j2))
                    if dist <= rg:
                        img[i1][i2] = 1
    return img 

def transform_img(img):
    u, v=img.shape

    def f(i,j):
        return i+0.1*np.sin(2*np.pi*j)
    def g(i,j):
        return j+0.1#*np.sin(3*np.pi*i)
    
    M , N =[], []
    for i in range(u):
        for j in range(v):
            i0, j0 = i/u, j/v
            u0=int(f(i0,j0)*300)
            v0=int(g(i0,j0)*300)
            M.append(u0)
            N.append(v0)

    m1,m2=max(M),max(N)
    n1,n2=min(M),min(N)
    oup=np.zeros((m1-n1,m2-n2))

    for i in range(u):
        for j in range(v):
            i0=i/u
            j0=j/v
            u0=int(f(i0,j0)*300)-n1-1
            v0=int(g(i0,j0)*300)-n2-1
            oup[u0,v0]=img[i,j]
    return interp_img(oup, img.shape)

def interp_img(a, size):
    h, w = size
    h0, w0 = a.shape
    y0 = np.linspace(0, h0-1, h0)
    x0 = np.linspace(0, w0-1, w0)
    f = interp2d(x0, y0, a)
    y = np.linspace(0, h0-1, h)
    x = np.linspace(0, w0-1, w)
    return f(x, y)

def remove_hrzs_near_fault(hrzs_img, fals_img):
    h, w = hrzs_img.shape
    hrzs_img_new = hrzs_img.copy()
    for i1 in range(h):
        for i2 in range(w):
            jj = np.arange(-1,1,1).tolist()
            kk = np.arange(-3,3,1).tolist()
            for j in jj:
                for k in kk:
                    ii1 = max(min(i1+j,h-1),0)
                    ii2 = max(min(i2+k,w-1),0)
                    if fals_img[ii1,ii2] == 1:
                        hrzs_img_new[i1,i2] =  0
    return hrzs_img_new

def find_sect_point(hrzs, fals, size):
    ilst = []
    n = size[-1]
    for hrz in hrzs:
        i1x,i2x = [],[]
        for i1s, i2s in hrz:
            i1x += i1s.tolist()
            i2x += i2s.tolist()
            
        f = interp1d(i2x, i1x, fill_value="extrapolate", kind='linear') 
        i1x, i2x = np.zeros(n), np.zeros(n)
        for k in range(n):
            i2x[k] = k
            i1x[k] = f(i2x[k]) 
        
        rds = []
        for i1z, i2z in fals:   
            mdist = 1e8 
            ed = 0
            for j in range(len(i1z)):     
                tmp = compute_edist((i1x,i2x), (i1z[j],i2z[j]))
                i = np.argmin(tmp)
                if tmp[i] < mdist:
                    mdist = tmp[i]
                    rd = i2x[i]
            rds.append(rd)
        ilst.append(rds)     
        
    return ilst

def find_near_list(a, n=1e1):
    a = np.array(a)
    b = np.around(a*n)
    c = np.unique(b)
    d = list()
    for i in range(len(c)):
        idxs = np.where(b==c[i])
        d.append(a[idxs].mean())
    return d

def compute_in_hrzs(fx, hms):          
    hrzs = []
    for hm in hms:
        fm = fx * hm
        i1s,i2s = [],[]
        n1,n2 = fm.shape
        for i2 in range(n2):
            tmp = np.where(fm[:,i2]>0)[0]
            if len(tmp):
                i1s.append(tmp.mean())
                i2s.append(i2)
        hrzs.append([i1s,i2s])
    
    hrzs = cut_lins_hors(hrzs)
    
    return hrzs

def compute_out_hrzs(rx, hvs):
    n1 = rx.shape[0]
    hds = extract_horizon_img2d(rx,hvs)
    
    hrzs = []
    for i,hd in enumerate(hds):
        n = len(hd)
        i1s,i2s = [],[]
        for i2 in range(n):
            if hd[i2]>0 and hd[i2]<n1-1:
                i1s.append(hd[i2])
                i2s.append(i2)
        hrzs.append([i1s,i2s])
    hrzs = cut_lins_hors(hrzs)    
    return hrzs

def cut_lins_hors(lines):
    hrzs = []
    for line in lines:
        if len(line[0]) == 0:
            continue
        line = np.array(line)
        idxs = np.argsort(line[1])
        i1s = line[0][idxs]
        i2s = line[1][idxs]
        ib = 0
        i = 0
        for i in range(len(i2s)-1):
            if i2s[i+1]>i2s[i]+1:
                hrzs.append([i1s[ib:i+1],i2s[ib:i+1]])
                ib = i+1
        hrzs.append([i1s[ib:i+1],i2s[ib:i+1]])
    return hrzs

def extract_horizon_img2d(ux,uv):
    if isinstance(uv,list):
        uv = np.array(uv)
    n1,n2 = ux.shape
    vx = np.zeros((len(uv),n2))
    for i2 in range(n2):
        c0 = ux[:,i2]
        for i,v in enumerate(uv):
            if v <= np.max(c0) and v >= np.min(c0):
                c0, x0 = np.unique(c0, return_index=True)
                i1 = np.argsort(c0, axis=0)
                x0 = np.take_along_axis(x0, i1, axis=0)
                c0 = np.take_along_axis(c0, i1, axis=0)
                c0 = gaussian_filter(c0, sigma=3)
                x0 = gaussian_filter(x0, sigma=3)
                f = interp1d(c0, x0, fill_value="extrapolate")
                vx[i,i2] = f(v)
    return vx

def separate_hrzs(fx, ux, bit, bit_rate):

    hrz_idxs = np.arange(0,bit-1,bit_rate)
    hrzs_g = []
    for hrz_idx in hrz_idxs[1:-1]:
        x, y = np.where((fx*(bit-1) >= hrz_idx - bit_rate/2) & (fx*(bit-1) < hrz_idx + bit_rate/2))
        if len(x):
            hrzs_g.append([x,y])
    hvs,hms = [],[]
    
    for i in range(len(hrzs_g)):
        x,y = hrzs_g[i]
        
        hm = np.zeros(fx.shape)
        for i in range(len(x)):
            hm[x[i]][y[i]] = 1.0     
        hv = np.sum(ux*hm)/hm.sum()
        hvs.append(hv)
        hms.append(hm)

    return hvs, hms  

def cut_lins_for_each_hrz(lines):
    hrzs = []
    for line in lines:
        hrz = []
        if len(line[0]) <= 1:
            continue
        line = np.array(line)
        idxs = np.argsort(line[1])
        i1s = line[0][idxs]
        i2s = line[1][idxs]
        ib = 0
        for i in range(len(i2s)-1):
            if i2s[i+1]>i2s[i]+4:
                hrz.append([i1s[ib:i+1],i2s[ib:i+1]])
                ib = i+1               
        hrz.append([i1s[ib:i+1],i2s[ib:i+1]])
        hrzs.append(hrz)
    return hrzs

def extract_hrzs_not_on_grid(rx, hvs):
    n1 = rx.shape[0]
    hds = extract_horizon_img2d(rx,hvs)
    
    hrzs = []
    for i,hd in enumerate(hds):
        n = len(hd)
        i1s,i2s = [],[]
        for i2 in range(n):
            if hd[i2]>0 and hd[i2]<n1-1:
                i1s.append(hd[i2])
                i2s.append(i2)
        hrzs.append([i1s,i2s])   
    return cut_lins_for_each_hrz(hrzs)

def compute_hrzs_not_on_grid(fx, hms):          
    hrzs = []
    for hm in hms:
        fm = fx * hm
        i1s,i2s = [],[]
        n1,n2 = fm.shape
        for i2 in range(n2):
            tmp = np.where(fm[:,i2]>0)[0]
            if len(tmp):
                i1s.append(tmp.mean())
                i2s.append(i2)
        hrzs.append([i1s,i2s])
    return cut_lins_for_each_hrz(hrzs)

def compute_unqiue_hrzs(frame):
    n1, n2 = frame.shape
    unique_list = np.unique(frame)[1:]
    hrzs = []
    for i in range(len(unique_list)//2):
        bg = unique_list[2*i]
        ed = unique_list[2*i+1]

        i1s, i2s = [], []
        for j in range(n2):
            tmp = []
            for k in range(n1):
                if frame[k,j] >= bg and frame[k,j] <= ed:
                    tmp.append(k)        
            if tmp:
                i2s.append(j)
                i1s.append(np.array(tmp).mean())
        hrzs.append([i1s,i2s])
    return hrzs 

def mask_horizons_func(z, x, hrz_grp=4, hrz_sel=None, sample_rate=1):
    if hrz_sel is None:
        hrz_sel = random.randint(hrz_grp//2, hrz_grp)
        
    hrz_t, hrz_b = np.min(z),np.max(z) 
    hrz_v = max((hrz_b - hrz_t) / hrz_grp, 1)
    hrz_ids = random.sample(list(range(hrz_grp)), hrz_sel)
            
    new_z = np.array([], dtype=np.int32)
    new_x = np.array([], dtype=np.int32)   
    for i in hrz_ids:   
        new_ids = np.where((z>=hrz_t+i*hrz_v) & (z < hrz_t+(i+1)*hrz_v))[0]                
        new_ids = random.sample(new_ids.tolist(), len(new_ids)//sample_rate)
        new_ids = np.array(new_ids, dtype=np.int32)  
        new_z = np.append(new_z, z[new_ids])
        new_x = np.append(new_x, x[new_ids])
   
    return new_z, new_x, hrz_ids

def get_train_sample_from_rgt(rgt, possible_num_hrzs, hrz_grp, 
                              bit=256, bit_mute=20, bit_rate=2, 
                              fl=None, fl_rg=2, 
                              orientation=None,
                              data_sample_path=None):
    
    if fl is not None:
        fl_padding = np.pad(fl, ((fl_rg,fl_rg),(fl_rg,fl_rg)), mode='reflect')
        
    num_hrzs = random.choice(possible_num_hrzs)
    
    ux = min_max_norm(rgt) * (bit - 1)
    
    valid_hrzs_idxs = np.sort(np.unique(np.round(ux)))
    mute_min = np.where(valid_hrzs_idxs <= bit_mute)[0].max()
    mute_max = np.where(valid_hrzs_idxs > bit-bit_mute)[0].min() 
    valid_hrzs_idxs = valid_hrzs_idxs[mute_min:mute_max].tolist()
                        
    itv_js = len(valid_hrzs_idxs) // num_hrzs
    max_num_iter = 10
    
    fo = np.zeros((2, *(ux.shape))) if orientation is not None else None
    fx = np.zeros(ux.shape)
    mk = np.zeros(ux.shape)
    
    for k in range(num_hrzs):
        
        iter_count = 0
        while True: 
            
            hrzs_idx = random.choice(valid_hrzs_idxs[k*itv_js:(k+1)*itv_js])
            x, y = np.where((ux >= hrzs_idx - bit_rate/2) & (ux < (hrzs_idx + bit_rate/2)))

            # fault
            if fl is not None:
                x_f,y_f = [],[]
                for i in range(len(x)):         
                    if fl_padding[x[i]:x[i]+fl_rg*2, y[i]:y[i]+fl_rg*2].any() > 0:
                        continue        
                    x_f.append(x[i])
                    y_f.append(y[i])     
                x_f, y_f = np.array(x_f, dtype=np.int32), np.array(y_f, dtype=np.int32) 
            else:
                x_f, y_f = x, y  
                
            if len(x_f) == 0:
                iter_count += 1
                if iter_count > max_num_iter:
                    break
                else:
                    continue
            
            # mask
            x_h, y_h, idx_h = mask_horizons_func(x_f, y_f, hrz_grp=hrz_grp)  
            
            if len(x_h) == 0:
                iter_count += 1
                if iter_count > max_num_iter:
                    break
                else:
                    continue            
            else:
                break
                
        mk[x_h,y_h] = 1
        fx[x_h,y_h] = rgt[x_h,y_h]
        if orientation is not None:
            fo[:,x_h,y_h] = orientation[:,x_h,y_h]

    if orientation is not None:
        return fx, fo, mk
    return fx, mk

def compute_error(output, target):
    abs_diff = np.abs(output - target)
    mse = np.nanmean((np.power(abs_diff, 2)))
    rmse = math.sqrt(mse)                                                                                                                       
    mae = np.nanmean(abs_diff)
    absrel = np.nanmean(abs_diff / target)
    mrpd = np.nanmean(abs_diff / (np.abs(output) + np.abs(target))) * 2
    return mse, mae, absrel, mrpd
