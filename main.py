#%% Imports -------------------------------------------------------------------

import re
import time
import pickle
import tifffile
import warnings
import numpy as np
from skimage import io
from pathlib import Path

from joblib import Parallel, delayed 

# bdtools
from bdtools.models import UNet
from bdtools.nan import nan_filt
from bdtools.norm import norm_pct

# Skimage
from skimage.measure import label
from skimage.transform import rescale
from skimage.morphology import binary_dilation, remove_small_objects

# Scipy
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, filtfilt

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from PyQt5.QtCore import QTimer
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)

#%% Comments ------------------------------------------------------------------

'''
- Pulse intensity (Fluo. int. change (s-1)) should be adjusted with various fr
'''

#%% Inputs --------------------------------------------------------------------

# Procedure
run_extract = 0
run_process = 1
run_analyse = 1

# Parmeters

# process()



# analyse()
thresh_coeff = 1.1
min_size = 320
tps = [0, 300, 900, 1800, 2700]

#%% Initialize ----------------------------------------------------------------

# Paths
# data_path = Path("D:\local_Ding\data")
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Ding\data")
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

def process(path, lowpass=False):
           
    # Asymetric least square (asl) 
    def get_als(y, lam=1e7, p=0.001, niter=5): # Parameters (lam=1e7, p=0.001, niter=5)
        L = len(y)
        D = diags([1, -2, 1],[0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def _process(y):
        sub = y - get_als(y)
        grd = np.gradient(sub, axis=0)
        if lowpass:
            fs, cutoff, order = 1, 0.1, 2
            grd = butter_lowpass_filter(grd, cutoff, fs, order)
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
        flt = nan_filt(stk, mask=msk > 0, kernel_size=(1, 3, 3), iterations=1)
        
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
        thresh_coeff=1.0, 
        min_size=320, 
        tps=[0, 300, 900, 1800, 2700],
        ):
    
    # Nested function(s) ------------------------------------------------------

    def auto_thresh(grd, thresh_coeff=0.25, pct_high=95):
        med = np.nanmedian(grd)
        hgh = np.nanpercentile(grd, pct_high)
        thresh = med + ((hgh - med) * thresh_coeff)
        return thresh, med, hgh

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
        
        # Segment pulses
        thresh, med, hgh = auto_thresh(grd, thresh_coeff=thresh_coeff)
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
                valid = lvals == l
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
               
        def safe_nanmean(a, axis=1):
            if a.shape[axis] == 0:
                return np.full(a.shape[0], np.nan)
            if np.all(np.isnan(a), axis=axis).all():
                return np.full(a.shape[0], np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    return np.nanmean(a, axis=axis)
        
        tmax_cat, tmax_area_cat, tmax_ints_cat = [], [], []
        area_reg_avgCat, ints_reg_avgCat = [], []
        for tp in range(1, len(tpf)):
            valid = (tmax > tpf[tp - 1]) & (tmax <= tpf[tp])
            num_valid = np.count_nonzero(valid)
            tmax_cat.append(num_valid)
            tmax_area_cat.append(tmax_area[valid])
            tmax_ints_cat.append(tmax_ints[valid])
            area_avg = safe_nanmean(area_reg[:, valid], axis=1)
            ints_avg = safe_nanmean(ints_reg[:, valid], axis=1)
            area_reg_avgCat.append(area_avg)
            ints_reg_avgCat.append(ints_avg)
        
        # tmax_cat, tmax_area_cat, tmax_ints_cat = [], [], []
        # area_reg_avgCat, ints_reg_avgCat = [], []
        # for tp in range(1, len(tpf)):
        #     valid = (tmax > tpf[tp - 1]) & (tmax <= tpf[tp])
        #     tmax_cat.append(np.sum(valid))
        #     tmax_area_cat.append(tmax_area[valid])
        #     tmax_ints_cat.append(tmax_ints[valid])           
        #     area_reg_avgCat.append(np.nanmean(area_reg[:, valid], axis=1))
        #     ints_reg_avgCat.append(np.nanmean(ints_reg[:, valid], axis=1))
                                    
        for tp in range(1, len(tps)):
            tmax_cat[tp - 1] /= (tps[tp] - tps[tp - 1]) / (60 * fr)  # pulse per min
               
        # Fill data
        data = {
            
            # General
            "path"            : path,            # file path
            "name"            : path.name,       # file name
            "nT"              : nT,              # number of timepoints
            "nP"              : nP,              # number of pulses
            "total_area"      : total_area,      # total area (msk area)
            
            # Thresholding
            "thresh"          : thresh,          # threshold for grd pulses
            "med"             : med, 
            "hgh"             : hgh,
                         
            # Data
            "area"            : area,            # row = time, col = pulse, val = area
            "ints"            : ints,            # row = time, col = pulse, val = ints
            "area_nSum"       : area_nSum,       # row = % of total area covered by pulse
            "tmax"            : tmax,            # row = time of pulse max. area
            "tmax_area"       : tmax_area,       # row = area of pulse max. area
            "tmax_ints"       : tmax_ints,       # row = ints of pulse max. area
            "tmax_cat"        : tmax_cat,        # ...
            "tmax_area_cat"   : tmax_area_cat,   # ...
            "tmax_ints_cat"   : tmax_ints_cat,   # ...
            "area_reg"        : area_reg,        # ...
            "ints_reg"        : ints_reg,        # ...
            "area_reg_avgCat" : area_reg_avgCat, # ...
            "ints_reg_avgCat" : ints_reg_avgCat, # ...
            
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
    grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif"
    dat_path = data_path / f"{path.stem}_rf-{rf}_data.pkl"
    fig_path = data_path / f"{path.stem}_rf-{rf}_fig.png"
       
    # Load data
    grd = io.imread(grd_path)
    with open(str(dat_path), "rb") as f:
        data = pickle.load(f)

    # Initialize
    tpf = [tp * fr for tp in tps]
    cmap = plt.get_cmap("turbo", len(tps))
    hlabels = [f"{int(tps[tp-1])} - {int(tps[tp])}" for tp in range(1, len(tps))]
    vlabels = [f"{int(tps[tp-1])}\n{int(tps[tp])}" for tp in range(1, len(tps))]

    # Create figure
    fig = plt.figure(figsize=(4, 4), layout="tight")

    mpl.rcParams.update({
        
        "font.family": "Consolas",
        "font.size": 2,
        "axes.labelsize": 4,
        "axes.titlesize": 5,
        "axes.titlepad": 4,
        "legend.fontsize": 4,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "xtick.color": "black",
        "ytick.color": "black",
        
        "axes.linewidth"   : 0.5,  # Line width for the axes
        "xtick.major.width": 0.25,  # Line width for major x-ticks
        "ytick.major.width": 0.25,  # Line width for major y-ticks
        "xtick.minor.width": 0.25,  # Line width for minor x-ticks
        "ytick.minor.width": 0.25,  # Line width for minor y-ticks
        
        "savefig.dpi": 300,
        "savefig.transparent": False,
        
    })
    
    # Cumulative Pulse Area
    ax_area = fig.add_subplot(3, 1, 1)
    ax_area.set_title("Cumulative Pulse Area")
    dat_area = data["area_nSum"]
    ax_area.plot(dat_area, linewidth=0.5)
    for tp in range(1, len(tps)):
        ax_area.axvspan(tpf[tp - 1], tpf[tp], ymin=0, ymax=0.03,
                        facecolor=cmap(tp - 1), alpha=1)
    ax_area.set_ylabel("Cumulative Pulse Area (pixels)")
    ax_area.set_xlabel("Time (s)")
    ax_area.set_ylim(-0.02, 0.3)
    
    # Add vertical line
    vline = ax_area.axvline(x=0, color="red", linestyle="--", linewidth=1)
    fig.vline = vline

    # Pulse Segmentation
    ax_hist = fig.add_subplot(3, 2, 3)
    ax_hist.set_title("Int. Dist.")
    ax_hist.hist(grd.ravel(), bins=1000)
    ax_hist.set_xlim(data["med"] * 0.5, data["med"] * 1.5)
    
    print(data["med"])

    # Pulse Frequency
    ax_freq = fig.add_subplot(3, 2, 4)
    ax_freq.set_title("Pulse Frequency")
    dat_freq = data["tmax_cat"]
    for tp in range(1, len(tps)):
        ax_freq.bar(vlabels[tp - 1], dat_freq[tp - 1], color=cmap(tp - 1))
    ax_freq.set_ylabel("Pulse Number (min-1)")
    ax_freq.set_xlabel("Time Categories (s)")

    # Boxplots (Area)
    ax_area_box = fig.add_subplot(3, 3, 8)
    ax_area_box.set_title("Pulse Area (cat.)")
    dat_area_box = data["tmax_area_cat"]
    for tp in range(1, len(tps)):
        ax_area_box.boxplot(dat_area_box[tp - 1], positions=[tp], widths=0.6, showfliers=False)
    ax_area_box.set_xticks(np.arange(1, len(tps)))
    ax_area_box.set_xticklabels(vlabels)
    ax_area_box.set_ylabel("Pulse Area (pixels)")
    ax_area_box.set_xlabel("Time Categories (s)")

    # Boxplots (Intensity)
    ax_int_box = fig.add_subplot(3, 3, 9)
    ax_int_box.set_title("Pulse Intensity (cat.)")
    dat_int_box = data["tmax_ints_cat"]
    for tp in range(1, len(tps)):
        ax_int_box.boxplot(dat_int_box[tp - 1], positions=[tp], widths=0.6, showfliers=False)
    ax_int_box.set_xticks(np.arange(1, len(tps)))
    ax_int_box.set_xticklabels(vlabels)
    ax_int_box.set_ylabel("Fluo. Int. Change (s-1)")
    ax_int_box.set_xlabel("Time Categories (s)")

    # Save figure if needed (or simply return it)
    # plt.savefig(fig_path, format="png")

    return fig
    
#%% Function : display() ------------------------------------------------------

class Display:
    
    def __init__(self, paths):
        self.paths = paths
        if isinstance(self.paths, Path):
            self.paths = [self.paths]
        self.z = 0
        self.idx = 0
        self.init_data()
        self.init_viewer()

        # QTimer vline updates
        self.timer = QTimer()
        self.timer.setInterval(20)  # delay in ms
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_vline)

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
        dat_path = data_path / f"{path.stem}_rf-{rf}_data.pkl"
        
        # Load data
        self.stk = io.imread(stk_path)
        self.flt = io.imread(flt_path)
        self.sub = io.imread(sub_path)
        self.grd = io.imread(grd_path)
        self.out = io.imread(out_path)
        self.lbl = io.imread(lbl_path)
        with open(str(dat_path), "rb") as f:
            self.data = pickle.load(f)
            
    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Layers
        
        self.viewer.add_image(
            self.stk, name="stk", visible=1, 
            colormap="plasma", 
            contrast_limits=get_contrast_limits(self.stk),
            blending="additive", 
            )
        self.viewer.add_image(
            self.flt, name="flt", visible=0, 
            colormap="plasma", 
            contrast_limits=get_contrast_limits(self.flt),
            blending="additive", 
            )
        self.viewer.add_image(
            self.sub, name="sub", visible=0,
            colormap="plasma", 
            contrast_limits=get_contrast_limits(self.sub),
            blending="additive",
            )
        self.viewer.add_image(
            self.grd, name="grd", visible=0,
            colormap="plasma", 
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
        
        self.viewer.dims.set_point(0, self.z)
    
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
        self.rad_sub = QRadioButton("sub")
        self.rad_seg = QRadioButton("seg")
        self.rad_raw.setChecked(True)
        dsp_group_layout.addWidget(self.rad_raw)
        dsp_group_layout.addWidget(self.rad_sub)
        dsp_group_layout.addWidget(self.rad_seg)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_raw.toggled.connect(
            lambda checked: self.show_raw() if checked else None)
        self.rad_sub.toggled.connect(
            lambda checked: self.show_sub() if checked else None)
        self.rad_seg.toggled.connect(
            lambda checked: self.show_seg() if checked else None)
        
        # Create texts
        self.info_path = QLabel()
        self.info_path.setFont(QFont("Consolas"))
        self.info_path.setText(
            f"{self.paths[self.idx].name}"
            )
        self.info_vars = QLabel()
        self.info_vars.setFont(QFont("Consolas"))
        self.info_vars.setText(
            f"thresh (grd) = {self.data['thresh']:.6f}"
            )       
        self.info_shortcuts = QLabel()
        self.info_shortcuts.setFont(QFont("Consolas"))
        self.info_shortcuts.setText(
            "prev/next stack  : page down/up"
            )
        
        # Create plot
        self.figure = plot(self.paths[self.idx])
        self.canvas = FigureCanvas(self.figure)
                
        # Assemble layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_path)
        self.layout.addWidget(self.info_vars)
        self.layout.addWidget(self.info_shortcuts)
        self.layout.addWidget(self.canvas)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        self.viewer.dims.events.current_step.connect(self.on_slice_change)
        
        # Shortcuts

        @self.viewer.bind_key("PageDown", overwrite=True)
        def prev_stk_key(viewer):
            self.prev_stk()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_stk_key(viewer):
            self.next_stk()
    
    # Methods
    
    def on_slice_change(self, event):
        self.z = self.viewer.dims.current_step[0]
        self.timer.start()
    
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
        
    def update_plot(self):
        index = self.layout.indexOf(self.canvas)
        self.layout.removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.figure = plot(self.paths[self.idx])
        self.canvas = FigureCanvas(self.figure)
        self.layout.insertWidget(index, self.canvas)
        self.canvas.draw_idle()
        
    def update_vline(self):
        self.figure.vline.set_xdata([self.z, self.z])
        self.canvas.draw_idle()
                            
    def next_stk(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self.init_data()
            self.update_layers()
            self.update_text()
            self.update_plot()
            
    def prev_stk(self):
        if self.idx > 0:
            self.idx -= 1
            self.init_data()
            self.update_layers()
            self.update_text()
            self.update_plot()

    def show_raw(self):
        for name in self.viewer.layers:
            name = str(name)
            if "stk" in name:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.layers.selection.active = self.viewer.layers["stk"]
        
    def show_sub(self):
        for name in self.viewer.layers:
            name = str(name)
            if "sub" in name:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.layers.selection.active = self.viewer.layers["sub"]

    def show_seg(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["grd", "out"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.layers.selection.active = self.viewer.layers["grd"]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # for path in ome_paths: 
        
        # # Execute
        # extract(path)
        # process(path)
        # analyse(
        #     path, 
        #     thresh_coeff=thresh_coeff, 
        #     min_size=min_size, 
        #     tps=tps,
        #     )
        
#%% 

    idx = 4
    path = ome_paths[idx]
    
    # Execute
    process(path, lowpass=False)
    analyse(
        path, 
        thresh_coeff=thresh_coeff, 
        min_size=min_size, 
        tps=tps,
        )
    
    # rf, fr = get_info(path)
    # grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif"
    # grd = io.imread(grd_path)
        
    # Plot
    plot(ome_paths[idx])
    
    # # Display
    # Display(ome_paths[idx])

#%% 

    # def auto_thresh(grd, thresh_coeff=0.25, pct_high=95):
    #     med = np.nanmedian(grd)
    #     hgh = np.nanpercentile(grd, pct_high)
    #     return medn + ((phgh - medn) * thresh_coeff)
        

    # fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    
    # for i, (ax, path) in enumerate(zip(axes.ravel(), ome_paths)):
        
    #     name = path.stem
    #     rf, fr = get_info(path)
    #     grd_path = data_path / f"{path.stem}_rf-{rf}_grd.tif"
    #     grd = io.imread(grd_path)
    #     grd = norm_pct(grd, sample_fraction=0.1)
        
    #     mean = np.nanmean(grd)
    #     medn = np.nanmedian(grd)
    #     plow = np.nanpercentile(grd, 5)
    #     phgh = np.nanpercentile(grd, 95)
        
    #     thresh_coeff = 0.25
    #     thresh = medn + ((phgh - medn) * thresh_coeff)
        
    #     print(
    #         f"{name} : mean       = {mean:.6f}\n"
    #         f"{name} : median     = {medn:.6f}\n"
    #         f"{name} : plow/phigh = {plow:.6f}/{phgh:.6f}\n"
    #         )

    #     ax.hist(grd.ravel(), bins=1000)
    #     ax.axvline(x=plow, color="k", linewidth=0.5, linestyle=":")
    #     ax.axvline(x=medn, color="k", linewidth=0.5)
    #     ax.axvline(x=thresh, color="r", linewidth=0.5)
    #     ax.axvline(x=phgh, color="k", linewidth=0.5, linestyle=":")
    #     ax.set_xlim(medn * 0.5, medn * 1.5)
        