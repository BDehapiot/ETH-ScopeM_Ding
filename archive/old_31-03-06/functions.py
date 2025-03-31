#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Function : match_lst() ----------------------------------------------------

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