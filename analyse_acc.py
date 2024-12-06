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
        
        sub_tvals_acc = [get_acc(tval, mode="full") for tval in sub_tvals] 
        grd_tvals_acc = [get_acc(tval, mode="full") for tval in grd_tvals]
        sub_tvals_cat_acc, grd_tvals_cat_acc = [], []
        for tp in range(1, len(tps)):
            sub_tvals_cat_acc.append(
                [get_acc(tval, mode="full") for tval in sub_tvals_cat[tp - 1]])
            grd_tvals_cat_acc.append(
                [get_acc(tval, mode="full") for tval in grd_tvals_cat[tp - 1]])
        
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
            "sub_tvals_cat_acc" : sub_tvals_cat_acc, # acc of sub_tvals_acc
            "grd_tvals_cat_acc" : grd_tvals_cat_acc, # acc of grd_tvals_acc
            
            # Spatial
            "ddts"          : ddts,          # Dijkstra distance transforms of analysed pixels
            "dists"         : dists,         # sorted distances of ddts
            "sorts"         : sorts,         # sorting indexes
            "sub_svals"     : sub_svals,     # spatial values for sub analysed pixels
            "grd_svals"     : grd_svals,     # spatial values for grd analysed pixels
            "sub_svals_acc" : sub_svals_acc, # acc of sub_svals
            "grd_svals_acc" : grd_svals_acc, # acc of grd_svals
            
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

avg = "experiment" # "cluster" or "experiment"

if avg == "cluster":
    
    m_sub_tvals_acc, m_grd_tvals_acc = [], []
    for data in acc_data:
        m_sub_tvals_acc.append(
            [np.nanmean(dat["sub_tvals_acc"], axis=0) for dat in data])
        m_grd_tvals_acc.append(
            [np.nanmean(dat["grd_tvals_acc"], axis=0) for dat in data])

    sub_tvals_acc_avg = np.nanmean(np.stack(match_list(
        [dat for data in m_sub_tvals_acc for dat in data])), axis=0)
    grd_tvals_acc_avg = np.nanmean(np.stack(match_list(
        [dat for data in m_grd_tvals_acc for dat in data])), axis=0)
    sub_tvals_acc_std = np.nanstd(np.stack(match_list(
        [dat for data in m_sub_tvals_acc for dat in data])), axis=0)
    grd_tvals_acc_std = np.nanstd(np.stack(match_list(
        [dat for data in m_grd_tvals_acc for dat in data])), axis=0)

if avg == "experiment":
    
    m_sub_tvals_acc, m_grd_tvals_acc = [], []
    for data in acc_data:
        m_sub_tvals_acc.append(
            np.concatenate([dat["sub_tvals_acc"] for dat in data]).T)
        m_grd_tvals_acc.append(
            np.concatenate([dat["grd_tvals_acc"] for dat in data]).T)
    m_sub_tvals_acc = match_list(m_sub_tvals_acc)
    m_grd_tvals_acc = match_list(m_grd_tvals_acc)
        
    sub_tvals_acc_avg, grd_tvals_acc_avg = [], []
    sub_tvals_acc_std, grd_tvals_acc_std = [], []
    
    sub_tvals_acc_avg = np.nanmean(np.stack(
        [np.nanmean(data, axis=1) for data in m_sub_tvals_acc]), axis=0)
    sub_tvals_acc_std = np.nanstd(np.stack(
        [np.nanmean(data, axis=1) for data in m_sub_tvals_acc]), axis=0)
    grd_tvals_acc_avg = np.nanmean(np.stack(
        [np.nanmean(data, axis=1) for data in m_grd_tvals_acc]), axis=0)
    grd_tvals_acc_std = np.nanstd(np.stack(
        [np.nanmean(data, axis=1) for data in m_grd_tvals_acc]), axis=0)

if avg == "cluster":

    m_sub_tvals_cat_acc = [[] for _ in range(len(tps) - 1)]
    m_grd_tvals_cat_acc = [[] for _ in range(len(tps) - 1)]
    for tp in range(1, len(tps)):
        tmp_sub_cat, tmp_grd_cat = [], []
        for data in acc_data:
            tmp_sub_cat.append(np.concatenate(
                [dat["sub_tvals_cat_acc"][tp - 1] for dat in data]))
            tmp_grd_cat.append(np.concatenate(
                [dat["grd_tvals_cat_acc"][tp - 1] for dat in data]))
            tmp_sub_cat = match_list(tmp_sub_cat, axis=1)
            tmp_grd_cat = match_list(tmp_grd_cat, axis=1)
            m_sub_tvals_cat_acc[tp - 1].append(np.concatenate(tmp_sub_cat).T)
            m_grd_tvals_cat_acc[tp - 1].append(np.concatenate(tmp_grd_cat).T)
            m_sub_tvals_cat_acc[tp - 1] = match_list(
                m_sub_tvals_cat_acc[tp - 1], axis=0)
            m_grd_tvals_cat_acc[tp - 1] = match_list(
                m_grd_tvals_cat_acc[tp - 1], axis=0)
        m_sub_tvals_cat_acc[tp - 1] = np.hstack(m_sub_tvals_cat_acc[tp - 1])
        m_grd_tvals_cat_acc[tp - 1] = np.hstack(m_grd_tvals_cat_acc[tp - 1])
    m_sub_tvals_cat_acc = match_list(m_sub_tvals_cat_acc, axis=0)
    m_grd_tvals_cat_acc = match_list(m_grd_tvals_cat_acc, axis=0)
       
    sub_tvals_cat_acc_avg = [
        np.nanmean(data, axis=1) for data in m_sub_tvals_cat_acc]
    sub_tvals_cat_acc_std = [
        np.nanstd(data, axis=1) for data in m_sub_tvals_cat_acc]
    grd_tvals_cat_acc_avg = [
        np.nanmean(data, axis=1) for data in m_grd_tvals_cat_acc]
    grd_tvals_cat_acc_std = [
        np.nanstd(data, axis=1) for data in m_grd_tvals_cat_acc]    

if avg == "experiment":
    
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
    
    '''
    
    !!! Need to average the things by experiments !!!
    
    '''
    
    pass

#%%

# plt.figure(figsize=(8, 12))
# cmap = plt.get_cmap("turbo", len(tps))
# # hlabels = [f"{tps[tp-1]} - {tps[tp]}" for tp in range(1, len(tps))]
# # vlabels = [f"{tps[tp-1]}\n{tps[tp]}" for tp in range(1, len(tps))]

# plt.subplot(2, 1, 1)
# plt.title(f"Fluo. Int. ACC (time) \n {avg} averaging")
# data_avg = sub_tvals_acc_avg
# data_std = sub_tvals_acc_std
# data_x = np.arange(len(data_avg))
# plt.plot(data_x, data_avg)
# plt.fill_between(
#     data_x, data_avg - data_std, data_avg + data_std, 
#     alpha=0.3, color='blue',
#     )
# plt.ylabel("Correlation")
# plt.xlabel("Time (s)")

# plt.subplot(2, 1, 2)
# plt.title(f"Fluo. Int. ACC (time) \n {avg} averaging")
# for tp in range(1, len(tps)):
#     data_avg = sub_tvals_cat_acc_avg[tp - 1]
#     data_std = sub_tvals_cat_acc_std[tp - 1]
#     data_x = np.arange(len(data_avg))
#     plt.plot(data_x, data_avg, color=cmap(tp - 1))
#     # plt.fill_between(
#     #     data_x, data_avg - data_std, data_avg + data_std, 
#     #     alpha=0.1, color=cmap(tp - 1),
#     #     )
# plt.ylabel("Correlation")
# plt.xlabel("Time (s)")

# plt.tight_layout()
# plt.show()

#%%

# plt.subplot(2, 2, 2)
# plt.title(f"Fluo. Int. Change (s-1) ACC (time) \n {avg} averaging")
# data_avg = grd_tvals_acc_avg
# data_std = grd_tvals_acc_std
# data_x = np.arange(len(data_avg))
# plt.plot(data_x, data_avg)
# plt.fill_between(
#     data_x, data_avg - data_std, data_avg + data_std, 
#     alpha=0.3, color='blue',
#     )
# plt.ylabel("Correlation")
# plt.xlabel("Time (s)")

# plt.subplot(2, 2, 4)
# plt.title(f"Fluo. Int. Change (s-1) ACC (time) \n {avg} averaging")
# for tp in range(1, len(tps)):
#     data_avg = grd_tvals_cat_acc_avg[tp - 1]
#     data_std = grd_tvals_cat_acc_std[tp - 1]
#     data_x = np.arange(len(data_avg))
#     plt.plot(data_x, data_avg, color=cmap(tp - 1))
#     # plt.fill_between(
#     #     data_x, data_avg - data_std, data_avg + data_std, 
#     #     alpha=0.1, color=cmap(tp - 1),
#     #     )
# plt.ylabel("Correlation")
# plt.xlabel("Time (s)")
    
#%%

# lab = 1
# t0, t1 = 300, 2000
# sub_tresh = 1.0
# grd_tresh = 0.2

# # -----------------------------------------------------------------------------

# sub_svals_acc_pavg = []
# grd_svals_acc_pavg = []
# for t in range(data[lab]["nT"]):
#     sub_valid = sub[t, ...][data[lab]["ridxs"]] > sub_tresh
#     grd_valid = grd[t, ...][data[lab]["ridxs"]] > grd_tresh  
#     sub_svals_acc_pavg.append(np.nanmean(
#         np.stack(match_list(data[lab]["sub_svals_acc"][t], axis=0))[sub_valid], axis=0))
#     grd_svals_acc_pavg.append(np.nanmean(
#         np.stack(match_list(data[lab]["grd_svals_acc"][t], axis=0))[grd_valid], axis=0))
    
# # -----------------------------------------------------------------------------    
    
# sub_svals_acc_pavg_t0avg = np.nanmean(
#     np.stack(sub_svals_acc_pavg[t0:t1]), axis=0)
# grd_svals_acc_pavg_t0avg = np.nanmean(
#     np.stack(grd_svals_acc_pavg[t0:t1]), axis=0)
# sub_svals_acc_pavg_t1avg = np.nanmean(
#     np.stack(sub_svals_acc_pavg[t1:-1]), axis=0)
# grd_svals_acc_pavg_t1avg = np.nanmean(
#     np.stack(grd_svals_acc_pavg[t1:-1]), axis=0)

# # -----------------------------------------------------------------------------

# plt.figure(figsize=(12, 12))

# plt.subplot(2, 1, 1)
# plt.plot(sub_svals_acc_pavg_t0avg)
# plt.plot(sub_svals_acc_pavg_t1avg)

# plt.subplot(2, 1, 2)
# plt.plot(grd_svals_acc_pavg_t0avg)
# plt.plot(grd_svals_acc_pavg_t1avg)
