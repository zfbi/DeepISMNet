import os
import torch
import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure
from scipy.interpolate import griddata, interp1d

import seaborn as sns
import matplotlib as mpl
sns.set_context('poster',font_scale=.8)
mpl.rc('figure',figsize=(15,2))

def draw_img(img, msk=None, cmap="jet", method="bilinear", cmin=None, cmax=None, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(img,cmap=cmap, interpolation=method)
    if msk is not None:
        plt.imshow(msk, alpha=0.4, cmap='jet', interpolation='nearest')  
    plt.colorbar(fraction=0.023,pad=0.02) 
    plt.clim(vmin=cmin,vmax=cmax)
    
def get_cmap_fault():
    return [[0, 'rgb(255,255,255)'], [0.5, 'rgb(255,255,255)'], 
         [0.5, 'rgb(0,0,0)'], [1.0, 'rgb(0,0,0)']]

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
        rgb = list(map(lambda x:round(x*255.0), rgb))
        rgb = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        color_0 = [numb_pre, rgb]
        color_1 = [numb, rgb]
        numb_pre = numb
        colors.append(color_0)
        colors.append(color_1)
        
    return colors
    
def get_horizon_scalar(point_set_list, volume): 
    hrv = []
    for point_set in point_set_list:
        x, y, z = point_set['x'], point_set['y'], point_set['z']
        v = volume[z,y,x].mean()
        point_set['c'] = v * np.ones(len(x))
        hrv.append(v)   
    return hrv

def draw_slice_line_surf(volume, x_slices=None, y_slices=None, z_slices=None, 
                         cmap='jet',colors=None, 
                         clab=None, isofs=None, isovol=None, surfs=None, surfs2=None, cmin=None, cmax=None,
                         points=None, points2=None, isovol_remove=None, mute_edge=None,
                         smap='portland', smin=None, smax=None, 
                         fmap='jet', fmin=None, fmax=None,
                         lighting=None, lightposition=None, opacity=0.1,
                         ):
    
    if cmap == 'fault':
        cmap = get_cmap_fault()
    elif cmap == 'model':
        cmap = get_cmap_model()
    
    if smap == 'fault':
        smap = get_cmap_fault()
    elif smap == 'model':
        smap = get_cmap_model()    
   
    if fmap == 'fault':
        fmap = get_cmap_fault()
    elif fmap == 'model':
        fmap = get_cmap_model()
        
    if lighting is None:
        lighting = {"ambient":0.4, "roughness":0.4, "specular":1.4, "fresnel":0.4}
        
    if len(volume.shape) > 3:
        volume = volume.squeeze()
    nz, ny, nx = volume.shape
    
    if cmin is None or cmax is None: 
        _cmin, _cmax = np.min(volume), np.max(volume)
    else:
        _cmin, _cmax = cmin, cmax
    
    if clab is None:
        showscale = False
    else:
        showscale = True
    
    fig = go.Figure()  
    # surf
    if surfs is not None:
        for surf in surfs:
            xx,yy,zz = [],[],[]
            for ix in range(10,nx-10,2):
                for iy in range(10,ny-10,2):
                    if surf[iy][ix]>0 and surf[iy][ix]<nz-1:
                        xx.append(ix)
                        yy.append(iy)
                        zz.append(surf[iy][ix])
                
            obj = {}        
            obj.update({"type": "mesh3d",
                        "x": xx,
                        "y": yy,
                        "z": zz,
                        "fuck":0,
                        "color": "cyan",
                        "opacity": 0.5})         
            fig.add_trace(obj)
    
    def create_intensity_from_volume(x, y, z, volume, rg): 
        n1, n2, n3 = volume.shape
        
        volume_padding = np.pad(volume, ((rg,rg),(rg,rg),(rg,rg)), mode='reflect')
        v = np.zeros(len(x))
        for k in range(len(x)):
            points_list, values_list = [], []    
            intx, inty, intz = int(x[k]), int(y[k]), int(z[k])
            grid = np.arange(-rg,rg+1,1)
            ng = len(grid)
            for j1 in range(ng):
                for j2 in range(ng):
                    for j3 in range(ng):
                        points_list.append([intz + rg + grid[j1], 
                                       inty + rg + grid[j2], 
                                       intx + rg + grid[j3]])
                        values_list.append(volume_padding[intz + rg + grid[j1], 
                                       inty + rg + grid[j2], 
                                       intx + rg + grid[j3]])
            points_list, values_list = np.array(points_list), np.array(values_list)
            v[k] = griddata(points_list, values_list, (z[k], y[k], x[k]), 
                     method='linear')
        return v   
    
    def volume_padding(x, rg=2):
        return np.pad(x[:,rg:-rg,rg:-rg], ((0,0),(rg,rg),(rg,rg)), mode='reflect')
    
    def remove_surf_near_faults(verts, faces, faults, fr=2):
        _faces = []
        _faults = np.pad(faults,((fr,fr),(fr,fr),(fr,fr)),
                         mode='reflect')
        for i in range(faces.shape[0]):
            x = verts[faces[i]].mean(axis=0).astype(np.int)
            if not _faults[x[2]:x[2]+fr*2, 
                  x[1]:x[1]+fr*2,
                  x[0]:x[0]+fr*2].any() > 0:
                _faces.append(faces[i])
        return np.array(_faces)     

    def remove_surf_near_edges(verts, faces, sz, rg=6):
        _faces = []
        for i in range(faces.shape[0]):
            x = verts[faces[i]].mean(axis=0).astype(np.int)
            if x[1] > rg and x[1] < sz[0]-rg and x[0] > rg and x[0] < sz[1]-rg :
                _faces.append(faces[i])
        return np.array(_faces)    

    def remove_point_near_edges(point_set, sz, rg=6):
        point_set_new = {}
        keys = point_set.keys()
        for key in keys:
            point_set_new[key] = []
        
        for i in range(len(point_set['z'])):
            if point_set["y"][i] > rg and point_set["y"][i] < sz[0]-rg \
            and point_set["x"][i] > rg and point_set["x"][i] < sz[1]-rg : 
                for key in keys: 
                    point_set_new[key].append(point_set[key][i])

        return point_set_new   
    
    # iso-surf
    if isofs is not None:

        if fmin is None or fmax is None: 
            _fmin, _fmax = np.min(isovol), np.max(isovol)
        else:
            _fmin, _fmax = fmin, fmax
        
        cmap_model = get_cmap_model()
        
        for ic, isof in enumerate(isofs):
            obj = {}

            verts, faces, _, _ = measure.marching_cubes(volume_padding(isovol).transpose(2,1,0), 
                                                                   isof, step_size=1)
            if mute_edge is not None:
                faces = remove_surf_near_edges(verts, faces, [ny, nx], mute_edge)
            
            if isovol_remove is not None:
                faces = remove_surf_near_faults(verts, faces, isovol_remove)
        
            if colors is None: 
                isov = (isof - _fmin) / (_fmax - _fmin)
                for i in range(len(cmap_model)//2):
                    if isov >= cmap_model[2*i][0] and isov < cmap_model[2*i+1][0]:
                        _color = cmap_model[2*i][1]   
                intensity = None
            elif colors == "elevation":
                intensity = verts[:, 0]
                _color = None
                _fmin, _fmax = np.min(verts[:, 0]), np.max(verts[:, 0])
            elif colors == "intensity":
                _color = None
                intensity = create_intensity_from_volume(verts[:, 0], verts[:, 1], verts[:, 2], isovol, 2)  
            else:
                intensity = None
                _color = colors   
                
            obj.update({"type": "mesh3d",
                        "x": verts[:, 0],
                        "y": verts[:, 1],
                        "z": verts[:, 2],
                        "i": faces[:, 0],
                        "j": faces[:, 1],
                        "k": faces[:, 2],
                        "intensity": intensity,
                        "color": _color,
                        "colorscale": fmap,
                        "showscale": False,
                        "cmin": _fmin,
                        "cmax": _fmax,
                        "lighting":lighting,
                        "lightposition":lightposition,
                        "opacity": 1.0})
            fig.add_trace(obj)
    
    if points is not None:
        
        if smin is None or smax is None:
            _smin, _smax = 0, len(points)-1
        else:
            _smin, _smax = smin, smax
        
        for k, point in enumerate(points):
            obj = {}
            
            if mute_edge is not None:
                point = remove_point_near_edges(point, [ny, nx])
            
            if "c" not in point.keys():
                point['c'] = len(point['x']) * [k]
            
            obj.update({"type": "scatter3d",
                        "x": point['x'],
                        "y": point['y'],
                        "z": point['z'],
                        "mode":"markers",
                        "marker":{
                            "symbol":"cross",
                            "size":3.0,
                            "color":point['c'],
                            "colorscale":smap,
                            "cmin":_smin,
                            "cmax":_smax,
                            "opacity": 1.0,
                            },
                        "showlegend":False
                        })
            fig.add_trace(obj)
            
    if points2 is not None:
        _smin, _smax = 0, len(points2)
        for point2 in points2:
            obj = {}
            
            _color = 'rgb(139,0,0)'
            obj.update({"type": "scatter3d",
                        "x": point2['x'],
                        "y": point2['y'],
                        "z": point2['z'],
                        "mode":"markers",
                        "marker":{
                            "size":1.0,
                            "color":_color,
                            "colorscale":"jet",
                            "cmin":_smin,
                            "cmax":_smax,                            
                            "opacity": opacity,
                            },
                        "showlegend":False
                        })
            fig.add_trace(obj)

    # surf
    if surfs2 is not None:
        _smin, _smax = 0, len(surfs2) 
        for surf in surfs2:
            
            xx, yy, zz = surf["x"], surf["y"], surf["z"]

            if "v" in surf.keys():
                _color = surf["v"]
            
            obj = {}        
            obj.update({"type": "mesh3d",
                        "x": xx,
                        "y": yy,
                        "z": zz,
                        "color":"cyan",
                        "colorscale":smap,
                        "cmin":_smin,
                        "cmax":_smax,
                        "opacity": 0.5})         
            fig.add_trace(obj)            
            
    if x_slices is not None:
        # x-slice
        yy = np.arange(0, ny, 1)
        zz = np.arange(0, nz, 1)
        yy,zz = np.meshgrid(yy,zz)
        for x_slice in x_slices:
            xx = x_slice * np.ones((ny, nz)).T
            vv = volume[:,:,x_slice]
            fig.add_trace(go.Surface(
                z=zz,
                x=xx,
                y=yy,
                surfacecolor=vv,
                colorscale=cmap,
                cmin=_cmin, cmax=_cmax,
                showscale=showscale,
                connectgaps=False,
                colorbar={"title":clab, 
                          "title_side":'right',
                          "len": 0.8,
                          "thickness": 8,
                          "xanchor":"right"}))
    if y_slices is not None:
        # y-slice
        xx = np.arange(0, nx, 1)
        zz = np.arange(0, nz, 1)
        xx,zz = np.meshgrid(xx,zz)
        for y_slice in y_slices:
            yy = y_slice * np.ones((nx, nz)).T
            vv = volume[:,y_slice,:] 
            
            fig.add_trace(go.Surface(
                z=zz,
                x=xx,
                y=yy,
                surfacecolor=vv,
                colorscale=cmap,
                cmin=_cmin, cmax=_cmax,
                showscale=False))
            
    if z_slices is not None:
        # z-slice
        xx = np.arange(0, nx, 1)
        yy = np.arange(0, ny, 1)
        xx,yy = np.meshgrid(xx,yy)
        for z_slice in z_slices:
            zz = z_slice * np.ones((nx, ny)).T
            vv = volume[z_slice,:,:]
            fig.add_trace(go.Surface(
                z=zz,
                x=xx,
                y=yy,
                surfacecolor=vv,
                colorscale=cmap,
                cmin=_cmin, cmax=_cmax,
                showscale=False))

    fig.update_layout(
            height=500,
            width=800,
            scene = {
            "aspectmode":"data", # data
            "xaxis": {"title":"", "showticklabels":False},
            "yaxis": {"title":"", "showticklabels":False},
            "zaxis": {"title":"", "showticklabels":False, "autorange":'reversed'},
            'camera_eye': {"x": 1.25, "y": 0.5, "z": 1.0},
            'camera_up': {"x": 0, "y": 0, "z": 1.5},
            'camera_center': {"x": -0.1, "y": 0.025, "z": -0.2},
            },
            margin=dict(t=0, l=0, b=0))
    fig.show()