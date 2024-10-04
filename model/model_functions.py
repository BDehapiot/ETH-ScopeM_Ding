#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np
from skimage import io
import albumentations as A
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches

# Skimage
from skimage.segmentation import find_boundaries 

#%% Functions: ----------------------------------------------------------------

def split_idx(n, validation_split=0.2):
    val_n = int(n * validation_split)
    trn_n = n - val_n
    idx = np.arange(n)
    np.random.shuffle(idx)
    trn_idx = idx[:trn_n]
    val_idx = idx[trn_n:]
    return trn_idx, val_idx

def get_display():
    pass

#%% Function: preprocess() ----------------------------------------------------
   
def preprocess(
        train_path, 
        msk_suffix="", 
        msk_type="normal", 
        img_norm="global",
        patch_size=0, 
        patch_overlap=0,
        ):
    
    valid_types = ["normal", "edt", "bounds"]
    if msk_type not in valid_types:
        raise ValueError(
            f"Invalid value for msk_type: '{msk_type}'."
            f" Expected one of {valid_types}."
            )

    valid_norms = ["none", "global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
    
    # Nested function(s) ------------------------------------------------------
    
    def open_data(train_path, msk_suffix):
        imgs, msks = [], []
        tag = f"_mask{msk_suffix}"
        for path in train_path.iterdir():
            if tag in path.stem:
                img_name = path.name.replace(tag, "")
                imgs.append(io.imread(path.parent / img_name))
                msks.append(io.imread(path))
        imgs = np.stack(imgs)
        msks = np.stack(msks)
        return imgs, msks
    
    def normalize(arr, pct_low=0.01, pct_high=99.99):
        return norm_pct(norm_gcn(arr)) 
    
    def _preprocess(img, msk):
                
        if img_norm == "image":
            img = normalize(img)
        
        if msk_type == "normal":
            msk = msk > 0
        elif msk_type == "edt":
            msk = get_edt(msk, normalize="object", parallel=False)
        elif msk_type == "bounds":
            msk = find_boundaries(msk)           

        if patch_size > 0:
            img = extract_patches(img, patch_size, patch_overlap)
            msk = extract_patches(msk, patch_size, patch_overlap)

        return img, msk
        
    # Execute -----------------------------------------------------------------

    imgs, msks = open_data(train_path, msk_suffix)

    if normalize == "global":
        imgs = normalize(imgs)
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_preprocess)(img, msk)
        for img, msk in zip(imgs, msks)
        )    
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    if patch_size > 0:
        imgs = np.stack([arr for sublist in imgs for arr in sublist])
        msks = np.stack([arr for sublist in msks for arr in sublist])
    
    imgs = imgs.astype("float32")
    msks = msks.astype("float32")
    
    return imgs, msks

#%% Function: augment() -------------------------------------------------------

def augment(imgs, msks, iterations):
        
    if iterations <= imgs.shape[0]:
        warnings.warn(f"iterations ({iterations}) is less than n of images")
        
    # Nested function(s) ------------------------------------------------------
    
    def _augment(imgs, msks, operations):      
        idx = np.random.randint(0, len(imgs) - 1)
        outputs = operations(image=imgs[idx,...], mask=msks[idx,...])
        return outputs["image"], outputs["mask"]
    
    # Execute -----------------------------------------------------------------
    
    operations = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.5),
        ])
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_augment)(imgs, msks, operations)
        for i in range(iterations)
        )
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    return imgs, msks
