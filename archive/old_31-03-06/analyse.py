#%% Imports -------------------------------------------------------------------

import re
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
Exp1,2,3 : timepoint = 1s
Exp4     : timepoint = 250ms
    
1-300s     - 2mM Glucose,
301-900s   - 20mM Glucose,
901-1800s  - 20mM Glucose,
1801-2700s - 20mM Glucose + 100uM Mefloquine,

'''

#%% Inputs --------------------------------------------------------------------

# Parameters
thresh_coeff = 0.25
min_size = 320

# Paths
data_path = Path("D:\local_Ding\data")
img_paths = list(data_path.glob("*stk.tif"))
path_idx = 0

#%% Initialize ----------------------------------------------------------------

if path_idx == 3:
    tps = [0, 1200, 3600, 7200, 10800] # Experiment timepoints
else:
    tps = [0, 300, 900, 1800, 2700]

if isinstance(path_idx, int):
    img_paths = [img_paths[path_idx]]

#%% Functions -----------------------------------------------------------------

def get_pulse_data(path, thresh_coeff, min_size, tps):
    
    # Nested function(s) ------------------------------------------------------

    def get_pulse_seg(grd, thresh, min_size=min_size):
        pulse_msk, pulse_out = [], []
        for t, img in enumerate(grd):
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

    # Fetch rescaling factor (rf)
    match = re.search(r"rf-([-+]?\d*\.\d+|\d+)", path.name)
    if match: 
        rf = float(match.group(1)) 
    
    # Load data
    grd = io.imread(str(path).replace("stk", "grd"))
    
    # Initialize
    min_size = int(min_size * rf)
    thresh = np.nanpercentile(grd, 99.9) / thresh_coeff
    print(f"{thresh:.3f}")

    # Segment pulses
    pulse_msk, pulse_out, pulse_lbl = get_pulse_seg(
        grd, thresh, min_size=min_size)
        
    # Measure pulses
    nT = grd.shape[0] 
    nP = np.max(pulse_lbl) 
    total_area = np.sum(~np.isnan(grd[0, ...]))
    
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
        "total_area"       : total_area,      # total area (msk area)
        
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
    
    m_area_nSum = []
    m_tmax, m_tmax_area, m_tmax_ints = [], [], []
    m_tmax_cat, m_tmax_area_cat, m_tmax_ints_cat = [], [], []
    m_area_reg_avgCat, m_ints_reg_avgCat = [], []
    for data in pulse_data: 
        m_area_nSum.append(data["area_nSum"]) 
        m_tmax.append(data["tmax"]) 
        m_tmax_area.append(data["tmax_area"]) 
        m_tmax_ints.append(data["tmax_ints"]) 
        m_tmax_cat.append(data["tmax_cat"])
        m_tmax_area_cat.append(data["tmax_area_cat"])
        m_tmax_ints_cat.append(data["tmax_ints_cat"])
        m_area_reg_avgCat.append(data["area_reg_avgCat"])
        m_ints_reg_avgCat.append(data["ints_reg_avgCat"])
        
    area_nSum_avg = np.nanmean(np.stack(match_list(m_area_nSum)).T, axis=1)
    area_nSum_std = np.nanstd(np.stack(match_list(m_area_nSum)).T, axis=1)
    tmax_cct = np.concatenate(m_tmax)
    tmax_area_cct = np.concatenate(m_tmax_area)
    tmax_ints_cct = np.concatenate(m_tmax_ints)
    tmax_cat_avg = np.nanmean(np.vstack(m_tmax_cat), axis=0)
    tmax_cat_std = np.nanstd(np.vstack(m_tmax_cat), axis=0)
    tmax_area_cat_cct, tmax_ints_cat_cct = [], []
    area_reg_avgCat_avg, ints_reg_avgCat_avg = [], []
    area_reg_avgCat_std, ints_reg_avgCat_std = [], []
    for tp in range(1, len(tps)):
        tmax_area_cat_cct.append(
            np.concatenate([data[tp - 1] for data in m_tmax_area_cat]))
        tmax_ints_cat_cct.append(
            np.concatenate([data[tp - 1] for data in m_tmax_ints_cat]))
        area_reg_avgCat_avg.append(
            np.nanmean(match_list([data[tp - 1] for data in m_area_reg_avgCat]), axis=0))
        ints_reg_avgCat_avg.append(
            np.nanmean(match_list([data[tp - 1] for data in m_ints_reg_avgCat]), axis=0))
        area_reg_avgCat_std.append(
            np.nanstd(match_list([data[tp - 1] for data in m_area_reg_avgCat]), axis=0))
        ints_reg_avgCat_std.append(
            np.nanstd(match_list([data[tp - 1] for data in m_ints_reg_avgCat]), axis=0))

    pulse_data_merged = {
        
        "area_nSum_avg"       : area_nSum_avg,       # ...
        "area_nSum_std"       : area_nSum_std,       # ... 
        "tmax_cct"            : tmax_cct,            # ...
        "tmax_area_cct"       : tmax_area_cct,       # ...
        "tmax_ints_cct"       : tmax_ints_cct,       # ...
        "tmax_cat_avg"        : tmax_cat_avg,        # ...
        "tmax_cat_std"        : tmax_cat_std,        # ...
        "tmax_area_cat_cct"   : tmax_area_cat_cct,   # ...
        "tmax_ints_cat_cct"   : tmax_ints_cat_cct,   # ...
        "area_reg_avgCat_avg" : area_reg_avgCat_avg, # ...
        "ints_reg_avgCat_avg" : ints_reg_avgCat_avg, # ...
        "area_reg_avgCat_std" : area_reg_avgCat_std, # ...
        "ints_reg_avgCat_std" : ints_reg_avgCat_std, # ...
        
        }
    
    return pulse_data_merged

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    pulse_data = []
    for path in img_paths:  
    
        t0 = time.time()    
        print(path.name)
        
        pulse_data.append(get_pulse_data(path, thresh_coeff, min_size, tps))
    
        t1= time.time()
        print(f"runtime : {t1 - t0:.3f}")

    pulse_data_merged = merge_pulse_data(pulse_data)

    # Display
    if isinstance(path_idx, int):
        viewer = napari.Viewer()
        viewer.add_image(
            pulse_data[0]["grd"], contrast_limits=[-2, 2], colormap="twilight")
        viewer.add_image(
            pulse_data[0]["pulse_out"], blending="translucent", opacity=0.5)
        viewer.add_labels(
            pulse_data[0]["pulse_lbl"], blending="translucent")

#%% Plot ----------------------------------------------------------------------

plt.figure(figsize=(8, 12))
cmap = plt.get_cmap("turbo", len(tps))
hlabels = [f"{tps[tp-1]} - {tps[tp]}" for tp in range(1, len(tps))]
vlabels = [f"{tps[tp-1]}\n{tps[tp]}" for tp in range(1, len(tps))]

# Pulse Area
plt.subplot(4, 1, 1)
plt.title("Cumulative Pulse Area")
data = pulse_data_merged["area_nSum_avg"]
plt.plot(data)
for tp in range(1, len(tps)):
    plt.axvspan(
        tps[tp - 1], tps[tp], ymin=0, ymax=0.03,
        facecolor=cmap(tp - 1), alpha=1
        )
plt.ylabel("Cumulative Pulse Area (pixels)")
plt.xlabel("Time (s)")
plt.ylim(-0.02, 0.3)

# Pulse Frequency
plt.subplot(4, 3, 4)
plt.title("Pulse Frequency")
data_avg = pulse_data_merged["tmax_cat_avg"]
data_std = pulse_data_merged["tmax_cat_std"]
for tp in range(1, len(tps)):
    plt.bar(
        vlabels[tp - 1], 
        data_avg[tp - 1], 
        yerr=data_std[tp - 1],
        color=cmap(tp - 1),
        capsize=5,
        )
plt.ylabel("Pulse Number (min-1)")
plt.xlabel("Time Categories (s)")

# Boxplots (Area)
plt.subplot(4, 3, 5)
plt.title("Pulse Area (cat.)")
data = pulse_data_merged["tmax_area_cat_cct"]
for tp in range(1, len(tps)):
    plt.boxplot(data[tp - 1], positions=[tp], widths=0.6, showfliers=False)
plt.xticks(np.arange(1, len(tps)), vlabels)
plt.ylabel("Pulse Area (pixels)")
plt.xlabel("Time Categories (s)")

# Boxplots (Intensity)
plt.subplot(4, 3, 6)
plt.title("Pulse Intensity (cat.)")
data = pulse_data_merged["tmax_ints_cat_cct"]
for tp in range(1, len(tps)):
    plt.boxplot(data[tp - 1], positions=[tp], widths=0.6, showfliers=False)
plt.xticks(np.arange(1, len(tps)), vlabels)
plt.ylabel("Fluo. Int. Change (s-1)")
plt.xlabel("Time Categories (s)")

# Pulse Area (registered & categorized)
plt.subplot(4, 1, 3)
plt.title("Pulse Area (registered & categorized)")
data_avg = pulse_data_merged["area_reg_avgCat_avg"]
data_std = pulse_data_merged["area_reg_avgCat_std"]
data_x = np.arange(len(data_avg[0]))
for tp in range(1, len(tps)):
    plt.errorbar(
        data_x, data_avg[tp - 1], 
        yerr=data_std[tp - 1],
        label=hlabels[tp - 1], 
        color=cmap(tp - 1),
        fmt='o-', capsize=5,
        )
plt.ylabel("Pulse Area (pixels)")
plt.xlabel("Time (s)")
plt.legend() 

# Pulse Intensity (registered & categorized)
plt.subplot(4, 1, 4)
data_avg = pulse_data_merged["ints_reg_avgCat_avg"]
data_std = pulse_data_merged["ints_reg_avgCat_std"]
data_x = np.arange(len(data_avg[0]))
plt.title("Pulse Intensity (registered & categorized)")
for tp in range(1, len(tps)):
    plt.errorbar(
        data_x, data_avg[tp - 1],  
        yerr=data_std[tp - 1],
        label=hlabels[tp - 1], 
        color=cmap(tp - 1),
        fmt='o-', capsize=5,
        )
plt.ylabel("Fluo. Int. Change (s-1)")
plt.xlabel("Time (s)") 
plt.legend() 

plt.tight_layout()
plt.show()
