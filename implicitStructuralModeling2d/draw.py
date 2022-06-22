import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from skimage import measure
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage import gaussian_filter

from PIL import Image
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import *

def draw_quiver(uu, vv, mk=None, order=2, frame=None, rgt=None):
    
    if mk is not None:
        x, y = np.where(mk>0)
        index = np.random.choice(np.arange(0,len(x),1), len(x)//order)
        x, y = x[index], y[index]        
    else:
        nx, ny = uu.shape
        xx, yy = np.meshgrid(np.linspace(order,nx-1-order,nx//order), np.linspace(order,ny-1-order,ny//order))
        xx, yy = np.round(xx).astype(np.int64), np.round(yy).astype(np.int64)
        x, y = xx.reshape(-1), yy.reshape(-1)
        
    u, v = uu[x,y], vv[x,y]
    
    ax = plt.gca()
    if frame is not None:
        ax.imshow(frame, cmap='viridis', interpolation='nearest')
    if rgt is not None:
        ax.contour(rgt,np.linspace(np.min(rgt),np.max(rgt),20),cmap='jet',linewidths=1)
    ax.quiver(y,x,u,v, minshaft=1, minlength=1, headaxislength=4.5,
               width=0.004, headwidth=3, headlength=5, color="black", alpha=0.8)
    
    plt.show()    

def draw_img(img, msk=None, cmap="jet", interpolation="bilinear",
             cmin=None, cmax=None):
    plt.imshow(img,cmap=cmap, interpolation=interpolation)
    
    if msk is not None:
        plt.imshow(msk, alpha=0.4, cmap='jet', interpolation='nearest')
    
    if cmin is None:
        cmin = np.min(img)
    if cmax is None:
        cmax = np.max(img)     
        
    plt.clim(cmin,cmax)
    plt.colorbar(fraction=0.023,pad=0.02)

def separate_hrzs(fx, ux, bit, bit_rate):

    hrz_idxs = np.arange(0,bit-1,bit_rate)
    # 分离层位
    hrzs_g = []
    for hrz_idx in hrz_idxs[1:-1]:      
        x, y = np.where((fx*(bit-1) >= hrz_idx - bit_rate/2) & (fx*(bit-1) < hrz_idx + bit_rate/2))
        if len(x):
            hrzs_g.append([x,y])
    # 单层位遮挡
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
    
def get_cmap_model():
    
    text = '''private static Color[] getStrataColors(double alpha) {
        float a = (float)alpha;
        Color[] c = new Color[256];
        for (int i=0; i<256; i++) {
          if (i<8) {
            c[i] = new Color(1f,0f,0f,a);
          } else if (i<16) {
            c[i] = new Color(1f,0.5019608f,0f,a);
          } else if (i<24) {
            c[i] = new Color(1f,1f,0f,a);
          } else if (i<32) {
            c[i] = new Color(0f,1f,0f,a);
          } else if (i<40) {
            c[i] = new Color(0f,0.5019608f,0f,a);
          } else if (i<48) {
            c[i] = new Color(0f,0.2509804f,0f,a);
          } else if (i<56) {
            c[i] = new Color(0f,1f,1f,a);
          } else if (i<64) {
            c[i] = new Color(0f,0.5019608f,1f,a);
          } else if (i<72) {
            c[i] = new Color(0f,0f,1f,a);
          } else if (i<80) {
            c[i] = new Color(0f,0f,0.627451f,a);
          } else if (i<88) {
            c[i] = new Color(0f,0.5019608f,0.7529412f,a);
          } else if (i<96) {
            c[i] = new Color(1f,0.5019608f,0.5019608f,a);
          } else if (i<104) {
            c[i] = new Color(0.5019608f,0.5019608f,1f,a);
          } else if (i<112) {
            c[i] = new Color(0.5019608f,0f,1f,a);
          } else if (i<120) {
            c[i] = new Color(0.5019608f,0f,0.5019608f,a);
          } else if (i<128) {
            c[i] = new Color(1f,0.5019608f,1f,a);
          } else if (i<136) {
            c[i] = new Color(1f,0f,1f,a);
          } else if (i<144) {
            c[i] = new Color(0.5019608f,0.2509804f,0f,a);
          } else if (i<152) {
            c[i] = new Color(0.5019608f,0.5019608f,0.5019608f,a);
          } else if (i<160) {
            c[i] = new Color(0.7529412f,0.7529412f,0.7529412f,a);
          } else if (i<168) {
            c[i] = new Color(0.2509804f,0f,0.2509804f,a);
          } else if (i<176) {
            c[i] = new Color(0.90588236f,0.7294118f,0.19607843f,a);
          } else if (i<184) {
            c[i] = new Color(0.44313726f,0.58431375f,0.58431375f,a);
          } else if (i<192) {
            c[i] = new Color(0.5254902f,0.42352942f,0.4862745f,a);
          } else if (i<200) {
            c[i] = new Color(0.7176471f,0.54509807f,0.44313726f,a);
          } else if (i<208) {
            c[i] = new Color(0.5019608f,0.5019608f,0f,a);
          } else if (i<216) {
            c[i] = new Color(0.7529412f,0.7294118f,0.8784314f,a);
          } else if (i<224) {
            c[i] = new Color(0.61960787f,0.85882354f,0.9882353f,a);
          } else if (i<232) {
            c[i] = new Color(0.7372549f,0.25882354f,0.24705882f,a);
          } else if (i<240) {
            c[i] = new Color(0.8862745f,0.8509804f,0.627451f,a);
          } else if (i<248) {
            c[i] = new Color(0.60784316f,0.9411765f,0.7490196f,a);
          } else if (i<256) {
            c[i] = new Color(0.62352943f,0.79607844f,0.105882354f,a);
          }
        }
        return c;
      }'''    

    lines = text.split('\n')
    colors = []
    numb_pre  = 0
    for i in range(4,len(lines)-4,2):
        numb = lines[i].split('<')[-1].split(')')[0]
        numb = float(numb)/256.0

        line = lines[i+1].split('(')[-1].split(')')[0].split(',')[:3]    
              
        rgb = []
        for j in line:
            rgb.append(float(j[:-1]))
        colors.append((rgb[0],rgb[1],rgb[2]))
    
    return LinearSegmentedColormap.from_list('my_cmap', colors, N=len(colors))

def draw_samples(samples_list, attr_list, cmap=None, norm=None, colorbar=True, 
                 gray_fc=0.9, bit=256, bit_rate=2,
                 save_file=None):
    
    r, num = len(attr_list), len(samples_list)   
    clabels = list(attr_list)
    
    if cmap is None:
        cmap = dict()
        for key in attr_list:
            if key in ["fault", "mask"]:
                cmap[key] = "gray"
            elif key in ["scalar", "frame"]:
                cmap[key] = "viridis"
            elif key in ["pred"]:
                cmap[key] = get_cmap_model()
            else:
                cmap[key] = "jet"

    methods = []
    for key in attr_list:
        if key.split('_')[0] in ["frame", "scalar", "fault", "mask", "pred", "orientation"]:
            methods.append("nearest")
        else:
            methods.append("bilinear")

    extent = None

    fig, axs = plt.subplots(r, num, sharey=False, figsize=(17*num, 8*r))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=None)

    for j, key in enumerate(attr_list):
        if norm is not None:
            norm_ = mpl.colors.Normalize(vmin=norm[j][0], vmax=norm[j][1])
        else:
            norm_ = None
                    
        for i in range(num):     
               
            if key == "fitting":
                
                pred, scalar = samples_list[i]["pred"][0].copy(), samples_list[i]["scalar"][0].copy()            
                pred = min_max_norm(pred)
                scalar = pred * scalar.astype(np.bool_).astype(np.single)
                bg_img = np.ones(pred.shape) * gray_fc

                hvs, _ = separate_hrzs(scalar, pred, bit, bit_rate)
                hvs = find_near_list(hvs)    
                hrzs = compute_out_hrzs(pred, hvs)
                
                scalar[scalar == 0.0] = np.nan   
                
                if (r == 1) & (num > 1):  
                    im = axs[i].imshow(bg_img, cmap='gray', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                      
                    im = axs[i].imshow(scalar, cmap=cmap["scalar"], interpolation='nearest', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)      
                    for i1s,i2s in hrzs:  
                        axs[i].plot(i2s, i1s, 'k--', linewidth=2)                    

                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(bg_img, cmap='gray', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                       
                    im = axs[j].imshow(scalar, cmap=cmap["scalar"], interpolation='nearest', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                     
                    for i1s,i2s in hrzs:  
                        axs[j].plot(i2s, i1s, 'k--', linewidth=2)  
                        
                elif (r == 1) & (num == 1):
                    im = axs.imshow(bg_img, cmap='gray', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                      
                    im = axs.imshow(scalar, cmap=cmap["scalar"], interpolation='nearest', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                     
                    for i1s,i2s in hrzs:  
                        axs.plot(i2s, i1s, 'k--', linewidth=2)  
                        
                else:
                    im = axs[j, i].imshow(bg_img, cmap='gray', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                      
                    im = axs[j, i].imshow(scalar, cmap=cmap["scalar"], interpolation='nearest', vmin=0, vmax=1,
                                       aspect='auto', extent=extent, norm=norm_)                     
                    for i1s,i2s in hrzs:  
                        axs[j, i].plot(i2s, i1s, 'k--', linewidth=2)  

            else:
                section = samples_list[i][key] 
                if len(section.shape) == 3:
                    section = section[0]

                if (r == 1) & (num > 1):
                    im = axs[i].imshow(section, aspect='auto', extent=extent, cmap=cmap[key], 
                                       interpolation=methods[j], norm=norm_)

                elif (r > 1) & (num == 1):
                    im = axs[j].imshow(section, aspect='auto', extent=extent, cmap=cmap[key], 
                                       interpolation=methods[j], norm=norm_)   

                elif (r == 1) & (num == 1):
                    im = axs.imshow(section, aspect='auto', extent=extent, cmap=cmap[key], 
                                    interpolation=methods[j], norm=norm_)

                else:
                    im = axs[j, i].imshow(section, aspect='auto', extent=extent, cmap=cmap[key], 
                                          interpolation=methods[j], norm=norm_)

            if (r == 1) & (num > 1):
                axs[i].set_xlabel('X', fontsize=18)
                axs[i].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[i].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[i], pad=0.02)
            elif (r > 1) & (num == 1):
                axs[j].set_xlabel('X', fontsize=18)
                axs[j].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[j].set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs[j], pad=0.02)
            elif (r == 1) & (num == 1):
                axs.set_xlabel('X', fontsize=18)
                axs.set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs.set_title(f'{clabels[j]}', fontsize=20)
                if colorbar:
                    fig.colorbar(im, ax=axs, pad=0.02)
            else:
                axs[j, i].set_xlabel('X', fontsize=18)
                axs[j, i].set_ylabel('Y', fontsize=18)
                if clabels is not None:
                    axs[j, i].set_title(f'{clabels[j]}', fontsize=20)   
                if colorbar:
                    fig.colorbar(im, ax=axs[j, i], pad=0.02)
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()