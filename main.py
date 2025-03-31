#%% Imports -------------------------------------------------------------------

import re
import time
import pickle
import tifffile
import numpy as np
from skimage import io
from pathlib import Path

from joblib import Parallel, delayed 

# bdtools
from bdtools.models import UNet
from bdtools.nan import nan_filt
from bdtools.norm import norm_pct, norm_gcn

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
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)

#%% Comments ------------------------------------------------------------------

#%% Inputs --------------------------------------------------------------------

# Procedure
run_extract = 0
run_process = 0
run_analyse = 1

# Parmeters

# analyse()
thresh_coeff = 1.5
min_size = 320
tps = [0, 300, 900, 1800, 2700]

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")
data_remote_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Ding\data")
ome_paths = list(data_path.glob("*.ome"))
ome_remote_paths = list(data_remote_path.glob("*.ome"))

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
    def get_als(y, lam=1e7, p=0.001, niter=5):
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
        thresh_coeff=0.25, 
        min_size=320, 
        tps=[0, 300, 900, 1800, 2700],
        ):
    
    global data
    
    # Nested function(s) ------------------------------------------------------

    def auto_thresh(grd, thresh_coeff=thresh_coeff):
        med = np.nanmedian(grd)
        std = np.nanstd(grd)
        thresh = med + (std * thresh_coeff)
        return med, std, thresh
    
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
    
    def get_stats(data):    
        avg = np.mean(data)
        std = np.std(data)
        sem = std / np.sqrt(len(data))
        return {"avg" : avg, "std" : std, "sem" : sem}
    
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
        total_area = np.sum(~np.isnan(grd[0, ...])) / rf
        tpf = [int(tp * fr) for tp in tps]
    
        # Segment pulses
        med, std, thresh = auto_thresh(grd)
        lbl, out = segment_pulses(grd, thresh, min_size=min_size)
        
        # Measure pulses
        
        nT = grd.shape[0] 
        nP = np.max(lbl) 
        nan_area = np.full((nT, nP), np.nan)
        nan_ints = np.full((nT, nP), np.nan)
        tmax = np.full(nP, np.nan)
        area = np.full(nP, np.nan)
        ints = np.full(nP, np.nan)
        
        for t in range(nT):
            vals = grd[t, ...].ravel()
            lvals = lbl[t, ...].ravel()
            for l in np.unique(lvals)[1:]:
                valid = lvals == l
                nan_area[t, l - 1] = np.sum(valid) / rf
                nan_ints[t, l - 1] = np.mean(vals[valid])
        
        for i, l in enumerate(range(nP)):
            lbl_area = nan_area[:, l]
            lbl_ints = nan_ints[:, l]
            lbl_tmax = np.nanargmax(lbl_area)
            valid = ~np.isnan(lbl_area)
            tmax[i] = lbl_tmax
            area[i] = lbl_area[lbl_tmax]
            ints[i] = lbl_ints[lbl_tmax]
        tdur = np.sum(~np.isnan(nan_area), axis=0) / fr
        acum = np.nansum(nan_area, axis=1) / total_area 
        
        tmax_cat, area_cat, ints_cat, tdur_cat, acum_cat = [], [], [], [], []
        for tp in range(1, len(tpf)):
            valid = (tmax > tpf[tp - 1]) & (tmax <= tpf[tp])
            num_valid = np.count_nonzero(valid)
            tmax_cat.append(num_valid)
            area_cat.append(area[valid])
            ints_cat.append(ints[valid])
            tdur_cat.append(tdur[valid])
            acum_cat.append(acum[tpf[tp - 1]:tpf[tp]])
            
        for tp in range(1, len(tps)):
            tmax_cat[tp - 1] /= (tps[tp] - tps[tp - 1]) / (60 * fr)  # pulse per min
            
        area_cat_stat, ints_cat_stat, tdur_cat_stat, acum_cat_stat = [], [], [], []
        for i in range(len(area_cat)):
            area_cat_stat.append(get_stats(area_cat[i]))
            ints_cat_stat.append(get_stats(ints_cat[i]))
            tdur_cat_stat.append(get_stats(tdur_cat[i]))
            acum_cat_stat.append(get_stats(acum_cat[i]))
                       
        # Fill data
        data = {
            
            # General
            
            "path"          : path,          # file path
            "name"          : path.name,     # file name
            "rf"            : rf,            # rescaling factor
            "fr"            : fr,            # frame rate (Hz)
            "tps"           : tps,           # time categories (seconds)
            "tpf"           : tpf,           # time categories (frames)
            "nT"            : nT,            # number of timepoints
            "nP"            : nP,            # number of pulses
            "thresh"        : thresh,        # threshold for grd pulses
                         
            # Data
            
            "nan_area"      : nan_area,      # row = time, col = pulse, val = area
            "nan_ints"      : nan_ints,      # row = time, col = pulse, val = ints
            
            "tmax"          : tmax,          # time of pulse max. area
            "area"          : area,          # pulse area at tmax
            "ints"          : ints,          # pulse intensity at tmax
            "tdur"          : tdur,          # pulse duration
            "acum"          : acum,          # cumulative pulse area
            
            "tmax_cat"      : tmax_cat,      # time cat. of tmax
            "area_cat"      : area_cat,      # time cat. of area
            "ints_cat"      : ints_cat,      # time cat. of ints
            "tdur_cat"      : tdur_cat,      # time cat. of tdur
            "acum_cat"      : acum_cat,      # time cat. of acum
            
            "area_cat_stat" : area_cat_stat, # stat of area_cat
            "ints_cat_stat" : ints_cat_stat, # stat of ints_cat
            "tdur_cat_stat" : tdur_cat_stat, # stat of tdur_cat
            "acum_cat_stat" : acum_cat_stat, # stat of acum_cat
            
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
    fig_path = data_path / f"{path.stem}_rf-{rf}_fig.png"
       
    # Load data
    with open(str(dat_path), "rb") as f:
        data = pickle.load(f)
    
    # Fetch
    tps, tpf = data["tps"], data["tpf"]
    
    # Initialize
    cmap = plt.get_cmap("turbo", len(tps))
    vlabels = [f"{int(tps[tp-1])}\n{int(tps[tp])}" for tp in range(1, len(tps))]
    
    # Create figure
    
    fig = plt.figure(figsize=(3, 3), layout="tight")
    gs = GridSpec(2, 3, figure=fig)
    
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
    
    # Top row -----------------------------------------------------------------
    
    # Cumulative Pulse Area
    ax_acum = fig.add_subplot(gs[0, :2]) 
    ax_acum.set_title("Cumulative Pulse Area")
    dat_acum = data["acum"]
    ax_acum.plot(dat_acum, linewidth=0.5)
    for tp in range(1, len(tps)):
        ax_acum.axvspan(tpf[tp - 1], tpf[tp], ymin=0, ymax=0.03,
                        facecolor=cmap(tp - 1), alpha=1)
    ax_acum.set_ylabel("Cumulative Pulse Area (pixels)")
    ax_acum.set_xlabel("Time (s)")
    ax_acum.set_ylim(-0.02, 0.3)
    ax_acum.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x / fr)}")
        )
        
    # Pulse Frequency
    ax_freq = fig.add_subplot(gs[0, 2]) 
    ax_freq.set_title("Pulse Frequency")
    dat_freq = data["tmax_cat"]
    for tp in range(1, len(tps)):
        ax_freq.bar(vlabels[tp - 1], dat_freq[tp - 1], color=cmap(tp - 1))
    ax_freq.set_ylabel("Pulse Number (min-1)")
    ax_freq.set_xlabel("Time Categories (s)")
    
    # Bottom row --------------------------------------------------------------    
        
    # Area
    ax_area = fig.add_subplot(gs[1, 0]) 
    ax_area.set_title("Pulse Area (cat.)")
    dat_area = data["area_cat_stat"]
    for tp in range(1, len(tps)):
        ax_area.bar(
            vlabels[tp - 1], dat_area[tp - 1]["avg"], 
            yerr=dat_area[tp - 1]["sem"],
            capsize=2, color=cmap(tp - 1),
            error_kw={'elinewidth': 0.5, 'capthick': 0.5}
            )
    ax_area.set_ylabel("Pulse Area (pixels)")
    ax_area.set_xlabel("Time Categories (s)")
    ax_area.set_ylim(0, 3000)
    
    # Duration
    ax_tdur = fig.add_subplot(gs[1, 1]) 
    ax_tdur.set_title("Pulse Duration (cat.)")
    dat_tdur = data["tdur_cat_stat"]
    for tp in range(1, len(tps)):
        ax_tdur.bar(
            vlabels[tp - 1], dat_tdur[tp - 1]["avg"], 
            yerr=dat_tdur[tp - 1]["sem"],
            capsize=2, color=cmap(tp - 1),
            error_kw={'elinewidth': 0.5, 'capthick': 0.5}
            )
    ax_tdur.set_ylabel("Pulse Duration (s-1)")
    ax_tdur.set_xlabel("Time Categories (s)")
    ax_tdur.set_ylim(0, 5)
    
    # Intensity
    ax_ints = fig.add_subplot(gs[1, 2]) 
    ax_ints.set_title("Pulse Intensity (cat.)")
    dat_int = data["ints_cat_stat"]
    for tp in range(1, len(tps)):
        ax_ints.bar(
            vlabels[tp - 1], dat_int[tp - 1]["avg"], 
            yerr=dat_int[tp - 1]["sem"],
            capsize=2, color=cmap(tp - 1),
            error_kw={'elinewidth': 0.5, 'capthick': 0.5}
            )
    ax_ints.set_ylabel("Fluo. Int. Change (s-1)")
    ax_ints.set_xlabel("Time Categories (s)")
    ax_ints.set_ylim(0, 1.0)

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
        self.rad_lbl = QRadioButton("lbl")
        self.rad_raw.setChecked(True)
        dsp_group_layout.addWidget(self.rad_raw)
        dsp_group_layout.addWidget(self.rad_sub)
        dsp_group_layout.addWidget(self.rad_seg)
        dsp_group_layout.addWidget(self.rad_lbl)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_raw.toggled.connect(
            lambda checked: self.show_raw() if checked else None)
        self.rad_sub.toggled.connect(
            lambda checked: self.show_sub() if checked else None)
        self.rad_seg.toggled.connect(
            lambda checked: self.show_seg() if checked else None)
        self.rad_lbl.toggled.connect(
            lambda checked: self.show_lbl() if checked else None)
        
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
        self.viewer.layers["grd"].colormap = "plasma"
        self.viewer.layers["grd"].opacity = 1.0
        self.viewer.layers.selection.active = self.viewer.layers["grd"]
        
    def show_lbl(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["grd", "out", "lbl"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.layers["grd"].colormap = "gray"
        self.viewer.layers["grd"].opacity = 0.25
        self.viewer.layers.selection.active = self.viewer.layers["lbl"]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Batch
    for path in ome_paths: 
        extract(path)
        process(path)
        analyse(
            path, 
            thresh_coeff=thresh_coeff, 
            min_size=min_size, 
            tps=tps,
            )
    Display(ome_paths)
    
    # # Selected
    # path = ome_paths[8]
    # extract(path)
    # process(path)
    # analyse(
    #     path, 
    #     thresh_coeff=thresh_coeff, 
    #     min_size=min_size, 
    #     tps=tps,
    #     )
    # Display(path)
