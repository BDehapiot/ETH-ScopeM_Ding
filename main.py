#%% Imports -------------------------------------------------------------------

import re
import time
import napari
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.nan import nan_filt
from bdtools.models import UNet

# Skimage
from skimage.measure import label
from skimage.transform import rescale
from skimage.morphology import binary_dilation, remove_small_objects

# Scipy
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve

#%% Inputs --------------------------------------------------------------------

# Procedure
run_extract = 0
run_process = 0
run_analyse = 0

# Parameters (analyse)
thresh_coeff = 0.1
min_size = 320
tps = [0, 300, 900, 1800, 2700]

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Ding\data")
ome_paths = list(data_path.glob("*.ome"))

#%% Function(s) ---------------------------------------------------------------

def match_list(lst, fill=np.nan, ref="max", direction="after", axis=0):
   
    matched_lst = lst.copy()

    if ref == "max":
        max_size = np.max([
            elm.shape[0] if elm.ndim == 1 else elm.shape[axis] for elm in lst])
        for i, elm in enumerate(lst):
            if elm.ndim == 1: # 1D arrays
                tmp_fill = np.full((max_size - elm.shape[0],), fill)
                if direction == "after":
                    matched_lst[i] = np.concatenate((elm, tmp_fill))
                elif direction == "before":
                    matched_lst[i] = np.concatenate((tmp_fill, elm))
            else: # 2D arrays
                if axis == 0:
                    tmp_fill = np.full(
                        (max_size - elm.shape[0], elm.shape[1]), fill)
                    if direction == "after":
                        matched_lst[i] = np.vstack((elm, tmp_fill))
                    elif direction == "before":
                        matched_lst[i] = np.vstack((tmp_fill, elm))
                elif axis == 1:
                    tmp_fill = np.full(
                        (elm.shape[0], max_size - elm.shape[1]), fill)
                    if direction == "after":
                        matched_lst[i] = np.hstack((elm, tmp_fill))
                    elif direction == "before":
                        matched_lst[i] = np.hstack((tmp_fill, elm))
    
    elif ref == "min":
        min_size = np.min([
            elm.shape[0] if elm.ndim == 1 else elm.shape[axis] for elm in lst])
        for i, elm in enumerate(lst):
            if elm.ndim == 1: # 1D arrays
                if direction == "after":
                    matched_lst[i] = elm[:min_size]
                elif direction == "before":
                    matched_lst[i] = elm[-min_size:]
            else: # 2D arrays
                if axis == 0:
                    if direction == "after":
                        matched_lst[i] = elm[:min_size, :]
                    elif direction == "before":
                        matched_lst[i] = elm[elm.shape[0] - min_size:, :]
                elif axis == 1:
                    if direction == "after":
                        matched_lst[i] = elm[:, :min_size]
                    elif direction == "before":
                        matched_lst[i] = elm[:, elm.shape[1] - min_size:]

    return matched_lst

#%% Function : extract() ------------------------------------------------------

def extract(path, rf):

    def _extract(img, rf):
        return rescale(img, rf, order=1, preserve_range=True)

    memmap = tifffile.memmap(str(path))
    stk = Parallel(n_jobs=-1)(
        delayed(_extract)(memmap[t,...], rf)
        for t in range(memmap.shape[0])
        )
    stk = np.stack(stk)
        
    return stk

#%% Function : process() ------------------------------------------------------

def process(path):
           
    # Asymetric least square (asl) 
    def get_als(y, lam=1e7, p=0.001, niter=5): # Parameters
        L = len(y)
        D = diags([1, -2, 1],[0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
    
    def _process(y):
        sub = y - get_als(y)
        grd = np.gradient(sub, axis=0)
        grd = np.roll(grd, 1, axis=0)
        return sub, grd
    
    # Fetch rescaling factor (rf)
    match = re.search(r"rf-([-+]?\d*\.\d+|\d+)", path.name)
    if match: 
        rf = float(match.group(1))     
    
    # Load data
    stk = io.imread(path)
    
    # Predict & create mask
    unet = UNet(load_name="model_64_normal_2000-440_1")
    prd = unet.predict(np.mean(stk, axis=0), verbose=0)
    msk = prd > 0.5
    msk = remove_small_objects(msk, min_size=2048*rf)
    
    # Filter stack
    flt = nan_filt(stk, mask=msk > 0, kernel_size=(1, 3, 3), iterations=3)
    
    # Temporal analysis
    sub = np.full_like(flt, np.nan)
    grd = np.full_like(flt, np.nan)
    for lab in np.unique(msk)[1:]:
        idxs = np.where(msk == lab)
        flt_tvals = flt[:, idxs[0], idxs[1]]
        outputs = Parallel(n_jobs=-1)(
            delayed(_process)(flt_tvals[:, i])
            for i in range(flt_tvals.shape[1])
            )
        sub_tvals = np.vstack([data[0] for data in outputs]).T
        grd_tvals = np.vstack([data[1] for data in outputs]).T
        sub[:, idxs[0], idxs[1]] = sub_tvals
        grd[:, idxs[0], idxs[1]] = grd_tvals
                   
    return flt, sub, grd

#%% Function : analyse() ------------------------------------------------------

def analyse(
        path, 
        thresh_coeff=0.1, 
        min_size=320, 
        tps=[0, 300, 900, 1800, 2700],
        display=False,
        ):
    
    # Nested function(s) ------------------------------------------------------

    def segment_pulses(grd, thresh, min_size=min_size):
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
    
    # Fetch exposure time (et)
    match = re.match(r".*_(\d{4})ms_.*", path.name)
    if match:
        et = int(match.group(1))   
    tps = [tp * (1000 / et) for tp in tps]
    
    # Load data
    grd = io.imread(str(path).replace("stk", "grd"))
    
    # Initialize
    min_size = int(min_size * rf)
    thresh = np.nanpercentile(grd, 99.9) * thresh_coeff

    # Segment pulses
    pulse_msk, pulse_out, pulse_lbl = segment_pulses(
        grd, thresh, min_size=min_size)
    
    # Dipslay
    if display:
        viewer = napari.Viewer()
        viewer.add_image(grd, colormap="twilight")
        viewer.add_image(pulse_out, blending="additive")
        return
        
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
            
    data = {
        
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
    
    return data

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob("*.ome"): 
        
        # Define rescaling factor acc. to binning
        rf = 0.1 if "b2" in path.name else 0.05 
        
        # Extract() -----------------------------------------------------------
        
        stk_path = data_path / f"{path.stem}_rf-{rf}_stk.tif"
        
        if not stk_path.exists() or run_extract:
            
            t0 = time.time()
            print(f"extract - {path.name} : ", end="", flush=True)
            stk = extract(path, rf)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
        
            # Save
            io.imsave(stk_path, stk.astype("float32"), check_contrast=False)
            
        # Process() -----------------------------------------------------------
        
        flt_path = data_path / f"{path.stem}_rf-{rf}_flt.tif"
        sub_path = data_path / f"{path.stem}_rf-{rf}_sub.tif"
        grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif"
        
        if not flt_path.exists() or run_process:
            
            t0 = time.time()
            print(f"process - {path.name} : ", end="", flush=True)
            flt, sub, grd = process(stk_path)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
        
            # Save
            io.imsave(flt_path, flt.astype("float32"), check_contrast=False)
            io.imsave(sub_path, sub.astype("float32"), check_contrast=False)
            io.imsave(grd_path, grd.astype("float32"), check_contrast=False)
            
        # Analyse() -----------------------------------------------------------
        
        # t0 = time.time()
        # print(f"analyse - {path.name} : ", end="", flush=True)
        # data = analyse(stk_path, thresh_coeff, min_size, tps)
        # t1 = time.time()
        # print(f"{t1 - t0:.3f}s")
        
#%%

    idx = 0
    path = list(data_path.glob("*.ome"))[idx]
    stk_path = data_path / f"{path.stem}_rf-{rf}_stk.tif"
    
    t0 = time.time()
    print(f"analyse - {path.name} : ", end="", flush=True)
    data = analyse(
        stk_path, 
        thresh_coeff=thresh_coeff, 
        min_size=min_size, 
        tps=tps,
        display=False,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    