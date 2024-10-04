#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

np.random.seed(42)

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")
train_path = Path(Path.cwd(), "train")

#%% Function(s): --------------------------------------------------------------

def get_indexes(nIdx, maxIdx):
    if maxIdx <= nIdx:
        idxs = np.arange(0, maxIdx)
    else:
        idxs = np.linspace(maxIdx / (nIdx + 1), maxIdx, nIdx, endpoint=False)
    idxs = np.round(idxs).astype(int)
    return idxs 

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    nIdx = 10

    for path in data_path.glob(f"*rf-{rf}_rstack.tif"):
        rstack = io.imread(path)
        idxs = get_indexes(nIdx, rstack.shape[0])
        for idx in idxs:
            io.imsave(
                train_path / (path.name.replace(".tif", f"_{idx:04d}.tif")),
                rstack[idx, ...], check_contrast=False,
                )    