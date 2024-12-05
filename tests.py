#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from functions import match_list

# Skimage
from skimage.measure import label
from skimage.morphology import binary_dilation, remove_small_objects

#%% Comments ------------------------------------------------------------------

'''
- currently the idea is to be able to plot with one movie or all implying 
optionnal averaging, sd & concatenating. 
'''

'''
# Experiment timepoints
1-300s     - 2mM Glucose,
301-900s   - 20mM Glucose,
901-1800s  - 20mM Glucose,
1801-2700s - 20mM Glucose + 100uM Mefloquine,
'''

#%% Inputs --------------------------------------------------------------------

# Parameters
rf = 0.05
thresh = 0.2
min_size = 16
tps = [0, 300, 900, 1800, 2700] # Experiment timepoints

# Paths
data_path = Path("D:\local_Ding\data")
img_paths = list(data_path.glob(f"*rf-{rf}_stk.tif*"))
path_idx = "all"

if isinstance(path_idx, int):
    img_paths = [img_paths[path_idx]]

#%% Functions -----------------------------------------------------------------

def get_pulse_data(path, thresh, min_size, tps):
    
    # Nested function(s) ------------------------------------------------------

    def get_pulse_seg(arr, thresh, min_size=min_size):
        pulse_msk, pulse_out = [], []
        for t, img in enumerate(arr):
            msk = img > thresh
            msk = remove_small_objects(msk, min_size=min_size)
            out = binary_dilation(msk) ^ msk
            pulse_msk.append(msk)
            pulse_out.append(out)
        pulse_msk = np.stack(pulse_msk)
        pulse_out = np.stack(pulse_out)
        pulse_lbl = label(pulse_msk)
        return pulse_msk, pulse_out, pulse_lbl
    
    # Execute -----------------------------------------------------------------    
    
    grd = io.imread(str(path).replace("stk", "grd"))
    
    pulse_msk, pulse_out, pulse_lbl = get_pulse_seg(
        grd, thresh, min_size=min_size)
        
    nT = grd.shape[0] # number of timepoints
    nP = np.max(pulse_lbl) # number of pulses
    total_area = np.sum(~np.isnan(grd[0, ...])) # total cells area (msk area)
    
    area = np.full((nT, nP), np.nan)
    ints = np.full((nT, nP), np.nan)
    for t in range(nT):
        vals = grd[t, ...].ravel()
        lvals = pulse_lbl[t, ...].ravel()
        for lbl in np.unique(lvals)[1:]:
            valid = lvals == lbl
            area[t, lbl - 1] = np.sum(valid)
            ints[t, lbl - 1] = np.mean(vals[valid])
    area_nSum = np.nansum(area, axis=1) / total_area
            
    tmax, tmax_area, tmax_ints = [], [], []  
    area_reg, ints_reg = [], []
    for lbl in range(nP):
        lbl_area = area[:, lbl]
        lbl_ints = ints[:, lbl]
        lbl_tmax = np.nanargmax(lbl_area)
        valid = ~np.isnan(lbl_area)
        tmax.append(lbl_tmax)
        tmax_area.append(lbl_area[lbl_tmax])
        tmax_ints.append(lbl_ints[lbl_tmax])
        area_reg.append(lbl_area[valid])
        ints_reg.append(lbl_ints[valid])
    tmax = np.stack(tmax)
    tmax_area = np.stack(tmax_area)
    tmax_ints = np.stack(tmax_ints)
    area_reg = np.vstack(match_list(area_reg)).T
    ints_reg = np.vstack(match_list(ints_reg)).T
    
    tmax_cat, tmax_area_cat, tmax_ints_cat = [], [], []
    area_reg_avgCat, ints_reg_avgCat = [], []
    for tp in range(1, len(tps)):
        valid = (tmax > tps[tp - 1]) & (tmax <= tps[tp])
        tmax_cat.append(np.sum(valid))
        tmax_area_cat.append(tmax_area[valid])
        tmax_ints_cat.append(tmax_ints[valid])
        area_reg_avgCat.append(np.nanmean(area_reg[:, valid], axis = 1))
        ints_reg_avgCat.append(np.nanmean(ints_reg[:, valid], axis = 1))
    
    for tp in range(1, len(tps)):
        tmax_cat[tp - 1] /= (tps[tp] - tps[tp - 1]) / 60 # pulse per min
            
    pulse_data = {
        
        # General
        "path"             : path,            # file path
        "name"             : path.name,       # file name
        "nT"               : nT,              # number of timepoints
        "nP"               : nP,              # number of pulses
        "total_area"       : total_area,      # total cells area (msk area)
        
        # Images
        "grd"              : grd,             # derivative of sub intensities
        "pulse_msk"        : pulse_msk,       # pulse mask
        "pulse_out"        : pulse_out,       # pulse outlines
        "pulse_lbl"        : pulse_lbl,       # pluse labels (3D)
        
        # Data
        "area"             : area,            # row = time, col = pulse, val = area
        "ints"             : ints,            # row = time, col = pulse, val = ints
        "area_nSum"        : area_nSum,       # row = % of total area covered by pulse
        "tmax"             : tmax,            # row = time of pulse max. area
        "tmax_area"        : tmax_area,       # row = area of pulse max. area
        "tmax_ints"        : tmax_ints,       # row = ints of pulse max. area
        "tmax_cat"         : tmax_cat,        # ...
        "tmax_area_cat"    : tmax_area_cat,   # ...
        "tmax_ints_cat"    : tmax_ints_cat,   # ...
        "area_reg"         : area_reg,        # ...
        "ints_reg"         : ints_reg,        # ...
        "area_reg_avgCat"  : area_reg_avgCat, # ...
        "ints_reg_avgCat"  : ints_reg_avgCat, # ...
        
        }
    
    return pulse_data

def merge_pulse_data(pulse_data):
    
    m_area_nSum, m_tmax, m_tmax_area, m_tmax_ints = [], [], [], []
    for data in pulse_data: 
        m_area_nSum.append(data["area_nSum"]) 
        m_tmax.append(data["tmax"]) 
        m_tmax_area.append(data["tmax_area"]) 
        m_tmax_ints.append(data["tmax_ints"]) 
        
    avg_area_nSum = np.nanmean(np.stack(match_list(m_area_nSum)).T, axis=1)
    std_area_nSum = np.nanstd(np.stack(match_list(m_area_nSum)).T, axis=1)
    cct_tmax = np.concatenate(m_tmax)
    cct_tmax_area = np.concatenate(m_tmax_area)
    cct_tmax_ints = np.concatenate(m_tmax_ints)
    
    m_pulse_data = {
                
        }
    
    return m_pulse_data

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    pulse_data = []
    for path in img_paths:  
    
        t0 = time.time()    
        print(path.name)
        
        pulse_data.append(get_pulse_data(path, thresh, min_size, tps))
    
        t1= time.time()
        print(f"runtime : {t1 - t0:.3f}")
                
    pulse_data = pulse_data[0] if len(pulse_data) == 1 else pulse_data
    
    # Display
    if isinstance(pulse_data, dict):
        viewer = napari.Viewer()
        viewer.add_image(pulse_data["grd"], contrast_limits=[-2, 2], colormap="twilight")
        viewer.add_image(pulse_data["pulse_out"], blending="translucent", opacity=0.5)
        viewer.add_labels(pulse_data["pulse_lbl"], blending="translucent")
    
    # Merge & process
    if isinstance(pulse_data, list): 
        pulse_data_merged = merge_pulse_data(pulse_data)
               
#%%



#%%

# m_tmax_cat = [
#     np.mean([data["tmax_cat"][tp - 1] for data in pulse_data])
#     for tp in range(1, len(tps))]
# m_tmax_cat_sd = [
#     np.std([data["tmax_cat"][tp - 1] for data in pulse_data])
#     for tp in range(1, len(tps))]
       
#%% Plot ----------------------------------------------------------------------

# plt.figure(figsize=(8, 12))
# cmap = plt.get_cmap("turbo", len(tps))
# hlabels = [f"{tps[tp-1]} - {tps[tp]}" for tp in range(1, len(tps))]
# vlabels = [f"{tps[tp-1]}\n{tps[tp]}" for tp in range(1, len(tps))]

# # Pulse Area
# plt.subplot(4, 1, 1)
# plt.title("Cumulative Pulse Area")
# data = np.nanmean(pulse_area, axis=1)
# plt.plot(data)
# for tp in range(1, len(tps)):
#     plt.axvspan(
#         tps[tp - 1], tps[tp], ymin=0, ymax=0.03,
#         facecolor=cmap(tp - 1), alpha=1
#         )
# plt.ylabel("Cumulative Pulse Area (pixels)")
# plt.xlabel("Time (s)")

# # Pulse Frequency
# plt.subplot(4, 3, 4)
# plt.title("Pulse Frequency")
# data = pulse_tmax_cat
# for tp in range(1, len(tps)):
#     plt.bar(vlabels[tp - 1], data[tp - 1], color=cmap(tp - 1))
# plt.ylabel("Pulse Number (min-1)")
# plt.xlabel("Time Categories (s)")

# # Boxplots (Area)
# plt.subplot(4, 3, 5)
# plt.title("Pulse Area (cat.)")
# data = pulse_tmax_area_cat
# for tp in range(1, len(tps)):
#     plt.boxplot(data[tp - 1], positions=[tp], widths=0.6, showfliers=False)
# plt.xticks(np.arange(1, 5), vlabels)
# plt.ylabel("Pulse Area (pixels)")
# plt.xlabel("Time Categories (s)")

# # Boxplots (Intensity)
# plt.subplot(4, 3, 6)
# plt.title("Pulse Intensity (cat.)")
# data = pulse_tmax_ints_cat
# for tp in range(1, len(tps)):
#     plt.boxplot(data[tp - 1], positions=[tp], widths=0.6, showfliers=False)
# plt.xticks(np.arange(1, 5), vlabels)
# plt.ylabel("Fluo. Int. Change (s-1)")
# plt.xlabel("Time Categories (s)")

# # Pulse Area (registered & categorized)
# plt.subplot(4, 1, 3)
# plt.title("Pulse Area (registered & categorized)")
# data = pulse_area_reg_avgCat
# for tp in range(1, len(tps)):
#     plt.plot(data[tp - 1], label=hlabels[tp - 1], color=cmap(tp - 1))
# plt.ylabel("Pulse Area (pixels)")
# plt.xlabel("Time (s)")
# plt.legend() 

# # Pulse Intensity (registered & categorized)
# plt.subplot(4, 1, 4)
# data = pulse_ints_reg_avgCat
# plt.title("Pulse Intensity (registered & categorized)")
# for tp in range(1, len(tps)):
#     plt.plot(data[tp - 1], label=hlabels[tp - 1], color=cmap(tp - 1))
# plt.ylabel("Fluo. Int. Change (s-1)")
# plt.xlabel("Time (s)") 
# plt.legend() 

# plt.tight_layout()
# plt.show()

#%%

# def process_pulse_data(pulse_data):
    
#     # Extract variables
#     tmax = pulse_data["tmax"]
#     tmax_area = pulse_data["tmax_area"]
#     tmax_ints = pulse_data["tmax_ints"]
#     area_reg = pulse_data["tmax_area"]
#     ints_reg = pulse_data["tmax_ints"]
    
#     tmax_cat, tmax_area_cat, tmax_ints_cat = [], [], []
#     area_reg_avgCat, ints_reg_avgCat = [], []
#     for tp in range(1, len(tps)):
#         valid = (tmax > tps[tp - 1]) & (tmax <= tps[tp])
#         tmax_cat.append(np.sum(valid))
#         tmax_area_cat.append(tmax_area[valid])
#         tmax_ints_cat.append(tmax_ints[valid])
#         area_reg_avgCat.append(np.nanmean(area_reg[:, valid], axis = 1))
#         ints_reg_avgCat.append(np.nanmean(ints_reg[:, valid], axis = 1))
    
#     for tp in range(1, len(tps)):
#         tmax_cat[tp - 1] /= (tps[tp] - tps[tp - 1]) / 60 # pulse per min

# area = np.hstack(match_list([data["area"] for data in pulse_data], axis=0))
# ints = np.hstack(match_list([data["ints"] for data in pulse_data], axis=0))
# tmax = np.concatenate([data["tmax"] for data in pulse_data])
# tmax_area = np.concatenate([data["tmax_area"] for data in pulse_data])
# tmax_ints = np.concatenate([data["tmax_ints"] for data in pulse_data])
# area_reg = np.hstack(match_list([data["area_reg"] for data in pulse_data], axis=0))
# ints_reg = np.hstack(match_list([data["ints_reg"] for data in pulse_data], axis=0))