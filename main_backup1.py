#%% Imports -------------------------------------------------------------------

import re
import time
import pickle
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
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

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

#%% Inputs --------------------------------------------------------------------

# Procedure
run_extract = 0
run_process = 0
run_analyse = 0

# Parameters (analyse)
thresh_coeff = 0.1
min_size = 320
tps = [0, 300, 900, 1800, 2700]
display = False

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Ding\data")
ome_paths = list(data_path.glob("*.ome"))

#%% Function(s) ---------------------------------------------------------------

def get_info(path):
    
    # Define rescaling factor (rf) acc. to binning
    rf = 0.1 if "b2" in path.name else 0.05 
    
    # Fetch frame rate (fr)
    match = re.match(r".*_(\d{4})ms.*", str(path.name))
    if match:
        exposure_time = int(match.group(1))  
    fr = 1000 / exposure_time
        
    return rf, fr

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

def extract(path):

    def _extract(img):
        return rescale(img, rf, order=1, preserve_range=True)

    # Get info
    rf, fr = get_info(path)

    # Path(s)
    stk_path = data_path / f"{path.stem}_rf-{rf}_stk.tif"

    if not stk_path.exists() or run_extract:
        
        t0 = time.time()
        print(f"extract - {path.name} : ", end="", flush=True)
        
        # Extract
        memmap = tifffile.memmap(str(path))
        stk = Parallel(n_jobs=-1)(
            delayed(_extract)(memmap[t,...], rf)
            for t in range(memmap.shape[0])
            )
        stk = np.stack(stk)
    
        # Save 
        io.imsave(stk_path, stk.astype("float32"), check_contrast=False)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

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
    
    # Get info
    rf, fr = get_info(path)
    
    # Path(s)
    stk_path = data_path / f"{path.stem}_rf-{rf}_stk.tif"
    flt_path = data_path / f"{path.stem}_rf-{rf}_flt.tif"
    sub_path = data_path / f"{path.stem}_rf-{rf}_sub.tif"
    grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif" 
    
    if not flt_path.exists() or run_process:
        
        t0 = time.time()
        print(f"process - {path.name} : ", end="", flush=True)
    
        # Load data
        stk = io.imread(stk_path)
        
        # Predict & create mask
        unet = UNet(load_name="model_64_normal_2000-440_1")
        prd = unet.predict(np.mean(stk, axis=0), verbose=0)
        msk = prd > 0.5
        msk = remove_small_objects(msk, min_size=2048*rf)
        
        # Filter stack
        flt = nan_filt(stk, mask=msk > 0, kernel_size=(1, 3, 3), iterations=3)
        
        # Process
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
            
        # Save
        io.imsave(flt_path, flt.astype("float32"), check_contrast=False)
        io.imsave(sub_path, sub.astype("float32"), check_contrast=False)
        io.imsave(grd_path, grd.astype("float32"), check_contrast=False)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Function : analyse() ------------------------------------------------------

def analyse(
        path,
        thresh_coeff=0.1, 
        min_size=320, 
        tps=[0, 300, 900, 1800, 2700],
        ):
    
    # Nested function(s) ------------------------------------------------------

    def segment_pulses(grd, thresh, min_size=min_size):
        lbl, out = [], []
        for t, img in enumerate(grd):
            msk = img > thresh
            msk = remove_small_objects(msk, min_size=min_size)
            lbl.append(msk)
            out.append(binary_dilation(msk) ^ msk)
        out = np.stack(out)
        lbl = np.stack(lbl)
        lbl = label(lbl)
        return lbl, out 
    
    # Execute -----------------------------------------------------------------    
    
    # Get info
    rf, fr = get_info(path)
    
    # Path(s)
    grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif"
    lbl_path = data_path / f"{path.stem}_rf-{rf}_lbl.tif"
    out_path = data_path / f"{path.stem}_rf-{rf}_out.tif"
    dat_path = data_path / f"{path.stem}_rf-{rf}_data.pkl"
    
    if not dat_path.exists() or run_analyse:

        t0 = time.time()
        print(f"analyse - {path.name} : ", end="", flush=True)        

        # Load data
        grd = io.imread(grd_path)
        
        # Initialize
        min_size = int(min_size * rf)
        tpf = [tp * fr for tp in tps]
        thresh = np.nanpercentile(grd, 99.9) * thresh_coeff   

        # Segment pulses
        lbl, out = segment_pulses(
            grd, thresh, min_size=min_size)
                    
        # Measure pulses
        nT = grd.shape[0] 
        nP = np.max(lbl) 
        total_area = np.sum(~np.isnan(grd[0, ...]))
        
        area = np.full((nT, nP), np.nan)
        ints = np.full((nT, nP), np.nan)
        for t in range(nT):
            vals = grd[t, ...].ravel()
            lvals = lbl[t, ...].ravel()
            for l in np.unique(lvals)[1:]:
                valid = lvals == lbl
                area[t, l - 1] = np.sum(valid)
                ints[t, l - 1] = np.mean(vals[valid])
        area_nSum = np.nansum(area, axis=1) / total_area
                
        tmax, tmax_area, tmax_ints = [], [], []  
        area_reg, ints_reg = [], []
        for l in range(nP):
            lbl_area = area[:, l]
            lbl_ints = ints[:, l]
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
        for tp in range(1, len(tpf)):
            valid = (tmax > tpf[tp - 1]) & (tmax <= tpf[tp])
            tmax_cat.append(np.sum(valid))
            tmax_area_cat.append(tmax_area[valid])
            tmax_ints_cat.append(tmax_ints[valid])
            area_reg_avgCat.append(np.nanmean(area_reg[:, valid], axis=1))
            ints_reg_avgCat.append(np.nanmean(ints_reg[:, valid], axis=1))
        
        for tp in range(1, len(tps)):
            tmax_cat[tp - 1] /= (tps[tp] - tps[tp - 1]) / (60 * fr)  # pulse per min
               
        # Fill data
        data = {
            
            # General
            "path"             : path,            # file path
            "name"             : path.name,       # file name
            "nT"               : nT,              # number of timepoints
            "nP"               : nP,              # number of pulses
            "total_area"       : total_area,      # total area (msk area)
                        
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
        
        # Save
        io.imsave(lbl_path, lbl.astype("uint16"), check_contrast=False)
        io.imsave(out_path, (out * 255).astype("uint8"), check_contrast=False)
        with open(str(dat_path), "wb") as f:
            pickle.dump(data, f)
            
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
    
#%% Function : plot() ---------------------------------------------------------

def plot(path, tps=[0, 300, 900, 1800, 2700]):
    
    # Get info
    rf, fr = get_info(path)
    
    # Path(s)
    dat_path = data_path / f"{path.stem}_rf-{rf}_data.pkl"
    
    t0 = time.time()
    print(f"analyse - {path.name} : ", end="", flush=True)  
    
    # Load data
    with open(str(dat_path), "rb") as f:
        data = pickle.load(f)
    
    # Initialize
    tpf = [tp * fr for tp in tps]
    
    # Plot
    plt.figure(figsize=(8, 12))
    cmap = plt.get_cmap("turbo", len(tps))
    hlabels = [
        f"{int(tps[tp-1])} - {int(tps[tp])}" 
        for tp in range(1, len(tps))
        ]
    vlabels = [
        f"{int(tps[tp-1])}\n{int(tps[tp])}" 
        for tp in range(1, len(tps))
        ]

    # Pulse Area
    plt.subplot(4, 1, 1)
    plt.title("Cumulative Pulse Area")
    dat = data["area_nSum"]
    plt.plot(dat)
    for tp in range(1, len(tps)):
        plt.axvspan(
            tpf[tp - 1], tpf[tp], ymin=0, ymax=0.03,
            facecolor=cmap(tp - 1), alpha=1
            )
    plt.ylabel("Cumulative Pulse Area (pixels)")
    plt.xlabel("Time (s)")
    plt.ylim(-0.02, 0.3)
    
    # Pulse Frequency
    plt.subplot(4, 3, 4)
    plt.title("Pulse Frequency")
    dat = data["tmax_cat"]
    for tp in range(1, len(tps)):
        plt.bar(
            vlabels[tp - 1], 
            dat[tp - 1], 
            color=cmap(tp - 1),
            )
    plt.ylabel("Pulse Number (min-1)")
    plt.xlabel("Time Categories (s)")
    
    # Boxplots (Area)
    plt.subplot(4, 3, 5)
    plt.title("Pulse Area (cat.)")
    dat = data["tmax_area_cat"]
    for tp in range(1, len(tps)):
        plt.boxplot(dat[tp - 1], positions=[tp], widths=0.6, showfliers=False)
    plt.xticks(np.arange(1, len(tps)), vlabels)
    plt.ylabel("Pulse Area (pixels)")
    plt.xlabel("Time Categories (s)")
    
    # Boxplots (Intensity)
    plt.subplot(4, 3, 6)
    plt.title("Pulse Intensity (cat.)")
    dat = data["tmax_ints_cat"]
    for tp in range(1, len(tps)):
        plt.boxplot(dat[tp - 1], positions=[tp], widths=0.6, showfliers=False)
    plt.xticks(np.arange(1, len(tps)), vlabels)
    plt.ylabel("Fluo. Int. Change (s-1)")
    plt.xlabel("Time Categories (s)")
    
    plt.tight_layout()
    plt.show()
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")    
    
#%% Function : display() ------------------------------------------------------

def get_contrast_limits(
        arr, 
        pct_low=0.1,
        pct_high=99.9,
        sample_fraction=0.01
        ):
    val = arr.ravel()
    val = np.random.choice(val, size=int(val.size * sample_fraction))
    pLow = np.nanpercentile(val, pct_low)
    pHigh = np.nanpercentile(val, pct_high)
    return [pLow, pHigh]

class Display:
    
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.init_data()
        self.init_viewer()

    def init_data(self):
        
        path = self.paths[self.idx]
        
        # Get info
        rf, fr = get_info(path)
        
        # Path(s)
        stk_path = data_path / f"{path.stem}_rf-{rf}_stk.tif"
        flt_path = data_path / f"{path.stem}_rf-{rf}_flt.tif"
        sub_path = data_path / f"{path.stem}_rf-{rf}_sub.tif"
        grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif" 
        out_path = data_path / f"{path.stem}_rf-{rf}_out.tif" 
        lbl_path = data_path / f"{path.stem}_rf-{rf}_lbl.tif"
        
        # Load data
        self.stk = io.imread(stk_path)
        self.flt = io.imread(flt_path)
        self.sub = io.imread(sub_path)
        self.grd = io.imread(grd_path)
        self.out = io.imread(out_path)
        self.lbl = io.imread(lbl_path)
    
    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Layers
        
        self.viewer.add_image(
            self.stk, name="stk", visible=1, 
            colormap="gray", 
            contrast_limits=get_contrast_limits(self.stk),
            blending="additive", 
            )
        self.viewer.add_image(
            self.flt, name="flt", visible=0, 
            colormap="gray", 
            contrast_limits=get_contrast_limits(self.flt),
            blending="additive", 
            )
        self.viewer.add_image(
            self.sub, name="sub", visible=0,
            colormap="gray", 
            contrast_limits=get_contrast_limits(self.sub),
            blending="additive",
            )
        self.viewer.add_image(
            self.grd, name="grd", visible=0,
            colormap="gray", 
            contrast_limits=get_contrast_limits(self.grd),
            blending="additive", 
            )
        self.viewer.add_image(
            self.out, name="out", visible=0,
            colormap="gray", 
            blending="additive",
            )
        self.viewer.add_labels(
            self.lbl, name="lbl", visible=0,
            blending="additive", 
            )
    
        # Create "stack" menu
        self.stk_group_box = QGroupBox("Select stack")
        stk_group_layout = QVBoxLayout()
        self.btn_next_stk = QPushButton("next")
        self.btn_prev_stk = QPushButton("prev")
        stk_group_layout.addWidget(self.btn_next_stk)
        stk_group_layout.addWidget(self.btn_prev_stk)
        self.stk_group_box.setLayout(stk_group_layout)
        self.btn_next_stk.clicked.connect(self.next_stk)
        self.btn_prev_stk.clicked.connect(self.prev_stk)
        
        # Create "display" menu
        self.dsp_group_box = QGroupBox("Display")
        dsp_group_layout = QHBoxLayout()
        self.rad_raw = QRadioButton("raw")
        self.rad_seg = QRadioButton("segmentation")
        self.rad_raw.setChecked(True)
        dsp_group_layout.addWidget(self.rad_raw)
        dsp_group_layout.addWidget(self.rad_seg)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_raw.toggled.connect(
            lambda checked: self.show_raw() if checked else None)
        self.rad_seg.toggled.connect(
            lambda checked: self.show_seg() if checked else None)
        
        # Create texts
        self.info_path = QLabel()
        self.info_path.setFont(QFont("Consolas"))
        self.info_path.setText(
            f"{self.paths[self.idx].name}"
            )
        self.info_shortcuts = QLabel()
        self.info_shortcuts.setFont(QFont("Consolas"))
        self.info_shortcuts.setText(
            "prev/next stack  : page down/up \n"
            )
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_path)
        self.layout.addWidget(self.info_shortcuts)
        
        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Shortcuts

        @self.viewer.bind_key("PageDown", overwrite=True)
        def prev_stk_key(viewer):
            self.prev_stk()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_stk_key(viewer):
            self.next_stk()
    
    # Methods
    
    def update_layers(self):
        self.viewer.layers["stk"].data = self.stk
        self.viewer.layers["stk"].contrast_limits = get_contrast_limits(self.stk)
        self.viewer.layers["flt"].data = self.flt
        self.viewer.layers["flt"].contrast_limits = get_contrast_limits(self.flt)
        self.viewer.layers["sub"].data = self.sub
        self.viewer.layers["sub"].contrast_limits = get_contrast_limits(self.sub)
        self.viewer.layers["grd"].data = self.grd
        self.viewer.layers["grd"].contrast_limits = get_contrast_limits(self.grd)
        self.viewer.layers["out"].data = self.out
        self.viewer.layers["lbl"].data = self.lbl
         
    def update_text(self):
        self.info_path.setText(f"{self.paths[self.idx].name}")
    
    def next_stk(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self.init_data()
            self.update_layers()
            self.update_text()
            
    def prev_stk(self):
        if self.idx > 0:
            self.idx -= 1
            self.init_data()
            self.update_layers()
            self.update_text()

    def show_raw(self):
        for name in self.viewer.layers:
            name = str(name)
            if "stk" in name:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0

    def show_seg(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["grd", "out"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in ome_paths: 
        
        # Execute
        extract(path)
        process(path)
        analyse(
            path, 
            thresh_coeff=thresh_coeff, 
            min_size=min_size, 
            tps=tps,
            )
    
    # Display
    Display(ome_paths)
    