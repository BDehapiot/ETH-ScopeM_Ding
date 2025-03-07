#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet
from bdtools.models.annotate import Annotate

#%% Inputs --------------------------------------------------------------------

# Procedure
extract = 0
annotate = 0
train = 1
predict = 0

# UNet build()
backbone = "resnet18"
activation = "sigmoid"

# UNet preprocess()
img_norm = "image"
msk_type = "normal"
patch_size = 64
patch_overlap = 32
downscaling_factor = 1

# UNet augment()
iterations = 2000
gamma_p = 0.5
gblur_p = 0.0 
noise_p = 0.5  
flip_p = 0.5
distord_p = 0.5 

# UNet train()
epochs = 100
batch_size = 32 
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0005
patience = 20

#%% Initialize ----------------------------------------------------------------

# Path
data_path = Path("D:\local_Ding\data")
train_path = Path("_remote", "train")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    if extract:
        
        for path in data_path.glob("*stk.tif"):
            stk = io.imread(path)
            prj = np.mean(stk, axis=0)
            prj = norm_pct(prj)
            io.imsave(
                train_path / str(path.name).replace("stk", "prj"),
                prj.astype("float32"), check_contrast=False,
                )
    
    if annotate:
        Annotate(train_path)
    
    if train:
        
        # Load data
        X, y = [], []
        for path in list(train_path.glob("*mask.tif")):
            y.append(io.imread(path))
            X.append(io.imread(str(path).replace("_mask", "")))
        X = np.stack(X)
        y = np.stack(y)
        X = np.tile(X, (3, 1, 1))
        y = np.tile(y, (3, 1, 1))
    
        # UNet build()
        unet = UNet(
            save_name="",
            load_name="",
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
        # UNet train()
        unet.train(
            
            X, y, 
            X_val=None, y_val=None,
            preview=False,
            
            # Preprocess
            img_norm=img_norm, 
            msk_type=msk_type, 
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            downscaling_factor=downscaling_factor, 
            
            # Augment
            iterations=iterations,
            gamma_p=gamma_p, 
            gblur_p=gblur_p, 
            noise_p=noise_p, 
            flip_p=flip_p, 
            distord_p=distord_p,
            
            # Train
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            metric=metric,
            learning_rate=learning_rate,
            patience=patience,
            
            )
    
    # Predict -----------------------------------------------------------------
    
    # if predict:
    #     unet = UNet(load_name="model_128_normal_2000-160_1")
    #     prd = unet.predict(rstk[::25], verbose=1)