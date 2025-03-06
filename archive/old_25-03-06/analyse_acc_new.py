#%% Imports -------------------------------------------------------------------

import time
import heapq
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

# Functions
from functions import match_list

# Skimage
from skimage.measure import label
from skimage.morphology import binary_dilation, remove_small_objects

#%% Comments ------------------------------------------------------------------

'''
1-300s - 2mM Glucose,
301-900 - 20mM Glucose,
901-1800 - 20mM Glucose
1801-2700 - 20mM Glucose + 100uM Mefloquine
'''

#%% Inputs --------------------------------------------------------------------

# Parameters
rf = 0.05
interval = 4
thresh = 0.2
min_size = 320 * rf # 32 for rf = 0.1
tps = [0, 300, 900, 2000, 2700] # Experiment timepoints

# Paths
data_path = Path("D:\local_Ding\data")
img_paths = list(data_path.glob(f"*rf-{rf}_stk.tif*"))
path_idx = "all"

if isinstance(path_idx, int):
    img_paths = [img_paths[path_idx]]

#%% Function(s): --------------------------------------------------------------

def analyse(path, interval, tps):

    # Nested function(s) ------------------------------------------------------
        
    # Reduce mask
    def reduce_msk(msk, interval):
        ys, xs = [], []
        idxs = np.where(msk)
        for y, x in zip(idxs[0], idxs[1]):
            if y % interval == 0 and x % interval == 0:
                ys.append(y)
                xs.append(x)
        ridxs = (ys, xs)
        rmsk = np.zeros_like(msk)
        rmsk[ridxs] = msk[ridxs]
        return rmsk
    
    # Auto cross correlation (acc) 
    def get_acc(y, mode="same"):
        acc = np.correlate(y, y, mode=mode)
        acc = acc[acc.size // 2:]
        acc /= acc[0] # zero lag normalization        
        return acc
    
    # Dijkstra distance transforms (ddts)  
    def get_ddts(msk, coord):

        # Define offsets
        s2 = 1.4142135623730951
        neighbor_offsets = [
            (-1, -1, s2), (-1, 0, 1), (-1, 1, s2),
            (0, -1, 1  ),             (0, 1, 1  ),
            (1, -1, s2 ), (1, 0, 1 ), ( 1, 1, s2),
            ]

        # Initialize
        cr, cc = coord
        nY, nX = msk.shape
        visited = np.zeros_like(msk, dtype=bool)
        ddts = np.full_like(msk, np.inf, dtype=float)
        ddts[cr, cc] = 0.0
        
        # Process
        heap = []
        heapq.heappush(heap, (0.0, (cr, cc)))
        while heap:
            current_dist, (r, c) = heapq.heappop(heap)
            if visited[r, c]:
                continue
            visited[r, c] = True
            for dr, dc, cost in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < nY and 0 <= nc < nX and
                        not visited[nr, nc] and msk[nr, nc] == 1):
                    new_dist = current_dist + cost
                    if new_dist < ddts[nr, nc]:
                        ddts[nr, nc] = new_dist
                        heapq.heappush(heap, (new_dist, (nr, nc)))

        # Replace infinite distances
        ddts[np.isinf(ddts)] = np.nan

        return ddts
    
    # Interpolated intensity profiles (iips)
    def process_svals(vals, sorts):
        svals, svals_acc = [], []
        vals = [vals[sort] for sort in sorts]
        for val, dist, xval in zip(vals, dists, xvals):
            sval = np.interp(xval, dist, val)
            sval_acc = get_acc(sval)
            svals.append(sval)
            svals_acc.append(sval_acc)
        return svals, svals_acc
    
    # Execute -----------------------------------------------------------------  

    msk = io.imread(str(path).replace("stk", "msk"))
    sub = io.imread(str(path).replace("stk", "sub")) 
    grd = io.imread(str(path).replace("stk", "grd"))
    
    acc_data = []
    rmsk = reduce_msk(msk, interval) 
    
    for lab in np.unique(rmsk)[1:]:
        
        ridxs = np.where(rmsk == lab)
        
        # Temporal analysis
        sub_tvals = list(sub[:, ridxs[0], ridxs[1]].T)
        grd_tvals = list(grd[:, ridxs[0], ridxs[1]].T)        

        sub_tvals_cat, grd_tvals_cat = [], []
        for tp in range(1, len(tps)):
            sub_tvals_cat.append(
                [tval[tps[tp - 1] : tps[tp]] for tval in sub_tvals])
            grd_tvals_cat.append(
                [tval[tps[tp - 1] : tps[tp]] for tval in grd_tvals])
        
        sub_tvals_acc = [get_acc(tval, mode="same") for tval in sub_tvals] 
        grd_tvals_acc = [get_acc(tval, mode="same") for tval in grd_tvals]
        sub_tvals_cat_acc, grd_tvals_cat_acc = [], []
        for tp in range(1, len(tps)):
            sub_tvals_cat_acc.append(
                [get_acc(tval, mode="same") for tval in sub_tvals_cat[tp - 1]])
            grd_tvals_cat_acc.append(
                [get_acc(tval, mode="same") for tval in grd_tvals_cat[tp - 1]])
        
        # Spatial analysis
        ddts = Parallel(n_jobs=-1)(
            delayed(get_ddts)(msk == lab, (y, x))
            for y, x in zip(ridxs[0], ridxs[1])
            )
        dists = [ddt[ridxs] for ddt in ddts]
        sorts = [np.argsort(dist) for dist in dists]  
        dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
        xvals = [np.arange(0, int(np.max(dist)), 1) for dist in dists]
        
        sub_svals, grd_svals = [], []  
        sub_svals_acc, grd_svals_acc = [], [] 
        for t in range(sub.shape[0]):
            svals, svals_acc = process_svals(sub[t, ...][ridxs], sorts)
            sub_svals.append(svals)
            sub_svals_acc.append(svals_acc)
            svals, svals_acc = process_svals(grd[t, ...][ridxs], sorts)
            grd_svals.append(svals)
            grd_svals_acc.append(svals_acc)
            
        sub_svals_cat_acc, grd_svals_cat_acc = [], []
        for tp in range(1, len(tps)):
            sub_svals_cat_acc.append(sub_svals_acc[tps[tp - 1] : tps[tp]])
            grd_svals_cat_acc.append(grd_svals_acc[tps[tp - 1] : tps[tp]])

        # Append data
        acc_data.append({
            
            "label" : lab,           # cluster label
            "ridxs" : ridxs,         # reduced mask indexes
            "nT"    : sub.shape[0],  # number of timepoints
            "nP"    : len(ridxs[0]), # number of analysed pixels
            
            # Temporal
            "sub_tvals"         : sub_tvals,         # temporal values for sub analysed pixels
            "grd_tvals"         : grd_tvals,         # temporal values for grd analysed pixels
            "sub_tvals_cat"     : sub_tvals_cat,     # categorized sub_tvals
            "grd_tvals_cat"     : grd_tvals_cat,     # categorized grd_tvals
            "sub_tvals_acc"     : sub_tvals_acc,     # acc of sub_tvals
            "grd_tvals_acc"     : grd_tvals_acc,     # acc of grd_tvals
            "sub_tvals_cat_acc" : sub_tvals_cat_acc, # acc of sub_tvals_cat
            "grd_tvals_cat_acc" : grd_tvals_cat_acc, # acc of grd_tvals_cat
            
            # Spatial
            "ddts"              : ddts,              # Dijkstra distance transforms of analysed pixels
            "dists"             : dists,             # sorted distances of ddts
            "sorts"             : sorts,             # sorting indexes
            "sub_svals"         : sub_svals,         # spatial values for sub analysed pixels
            "grd_svals"         : grd_svals,         # spatial values for grd analysed pixels
            "sub_svals_acc"     : sub_svals_acc,     # acc of sub_svals
            "grd_svals_acc"     : grd_svals_acc,     # acc of grd_svals
            "sub_svals_cat_acc" : sub_svals_cat_acc, # categorized acc of sub_svals
            "grd_svals_cat_acc" : grd_svals_cat_acc, # categorized acc of grd_svals
            
            })
    
    return acc_data

def merge_acc_data(acc_data):
    pass
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    acc_data = []
    for path in img_paths:  
                    
        t0 = time.time()    
        print(path.name)
        
        acc_data.append(analyse(path, interval, tps))
        
        t1= time.time()
        print(f"runtime : {t1 - t0:.3f}")
                  
#%%

# 
m_sub_tvals_acc, m_grd_tvals_acc = [], []
for data in acc_data:
    m_sub_tvals_acc.append(
        np.nanmean(np.concatenate([dat["sub_tvals_acc"] for dat in data]).T, axis=1))
    m_grd_tvals_acc.append(
        np.nanmean(np.concatenate([dat["grd_tvals_acc"] for dat in data]).T, axis=1))
m_sub_tvals_acc = match_list(m_sub_tvals_acc)
m_grd_tvals_acc = match_list(m_grd_tvals_acc)
sub_tvals_acc_avg = np.nanmean(np.stack(m_sub_tvals_acc), axis=0)
grd_tvals_acc_avg = np.nanmean(np.stack(m_grd_tvals_acc ), axis=0)
sub_tvals_acc_std = np.nanstd(np.stack(m_sub_tvals_acc), axis=0)
grd_tvals_acc_std = np.nanstd(np.stack(m_grd_tvals_acc ), axis=0)

#
m_sub_tvals_cat_acc, m_grd_tvals_cat_acc = [], []
for data in acc_data:
    tmp_sub_cat, tmp_grd_cat = [], []
    for tp in range(1, len(tps)):
        tmp_sub_cat.append(np.nanmean(np.concatenate(
            [dat["sub_tvals_cat_acc"][tp - 1] for dat in data]).T, axis=1))
        tmp_grd_cat.append(np.nanmean(np.concatenate(
            [dat["grd_tvals_cat_acc"][tp - 1] for dat in data]).T, axis=1))
    m_sub_tvals_cat_acc.append(tmp_sub_cat)
    m_grd_tvals_cat_acc.append(tmp_grd_cat)

sub_tvals_cat_acc_avg, grd_tvals_cat_acc_avg = [], []
sub_tvals_cat_acc_std, grd_tvals_cat_acc_std = [], []
for tp in range(1, len(tps)):
    sub_tvals_cat_acc_avg.append(np.nanmean(np.stack(match_list(
        [data[tp - 1] for data in m_sub_tvals_cat_acc])).T, axis=1))
    sub_tvals_cat_acc_std.append(np.nanstd(np.stack(match_list(
        [data[tp - 1] for data in m_sub_tvals_cat_acc])).T, axis=1))
    grd_tvals_cat_acc_avg.append(np.nanmean(np.stack(match_list(
        [data[tp - 1] for data in m_grd_tvals_cat_acc])).T, axis=1))
    grd_tvals_cat_acc_std.append(np.nanstd(np.stack(match_list(
        [data[tp - 1] for data in m_grd_tvals_cat_acc])).T, axis=1))

#%%

fig = plt.figure(figsize=(10, 10))
cmap = plt.get_cmap("turbo", len(tps))

gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# -----------------------------------------------------------------------------

data_sub_y = sub_tvals_acc_avg
data_sub_x = np.arange(0, len(data_sub_y))
data_grd_y = grd_tvals_acc_avg
data_grd_x = np.arange(0, len(data_grd_y))

# Plot 1
ax1.set_title("???")
ax1.axhline(y=0, color="k", linestyle='--', linewidth=0.5)
ax1.plot(data_sub_x, data_sub_y)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlim(-100, 2100)

# Plot 2
ax2.set_title("???")
ax2.axhline(y=0, color="k", linestyle='--', linewidth=0.5)
ax2.plot(data_grd_x, data_grd_y)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlim(-100, 2100)

# Plot 3
ax3.set_title("???")
ax3.axhline(y=0, color="k", linestyle='--', linewidth=0.5)
ax3.plot(data_grd_x, data_grd_y)
ax3.set_ylim(-0.05, 0.1)
ax3.set_xlim(0, 100)

# -----------------------------------------------------------------------------

data_grd_y = grd_tvals_cat_acc_avg
data_grd_x = [np.arange(0, len(data)) for data in data_grd_y]

# Plot 4
ax4.set_title("???")
ax4.axhline(y=0, color="k", linestyle='--', linewidth=0.5)
for tp in range(1, len(tps)):
    ax4.plot(data_grd_x[tp - 1], data_grd_y[tp - 1], color=cmap(tp - 1))
ax4.set_ylim(-0.05, 0.1)
ax4.set_xlim(0, 100)

# -----------------------------------------------------------------------------

plt.tight_layout()
plt.show()
