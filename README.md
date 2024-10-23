![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![TensorFlow Badge](https://img.shields.io/badge/TensoFlow-2.10-rgb(255%2C115%2C0)?logo=TensorFlow&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![CUDA Badge](https://img.shields.io/badge/CUDA-11.2-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![cuDNN Badge](https://img.shields.io/badge/cuDNN-8.1-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))    
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2024--10--03-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

# ETH-ScopeM_Ding  
Spatio-temporal calcium fluo. probe analysis

## Index
- [Installation](#installation)
- [Procedure](#procedure)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run one of the following command:  
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf_gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf_nogpu.yml
```  
- Activate Conda environment:
```bash
conda activate ding
```
Your prompt should now start with `(ding)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run one of the following command: 
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf_gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf_nogpu.yml
```  
- Activate Conda environment:  
```bash
conda activate ding
```
Your prompt should now start with `(ding)` instead of `(base)`

</details>


## Procedure

### `extract.py`
Read `.ome` image stacks from the `data_path` folder and perform:
1) **Spatial rescaling**  
Reduce spatial resolution, according to `rf` parameter to improve computing 
speed and signal to noise ratio. 
2) **Temporal roll-averaging**  
Rolling average over time according to `window_size` to reduce noise and 
facilitate later segmentation steps.

```bash
# Parameters
- data_path         # str, path to data directory
- rf                # float, rescaling factor (should be between 0 and 1)
- window_size       # int, temporal window size for rollling average (should be odd)

# Outputs
- ..._stack.tif     # float32 ndarray, rescaled stack
- ..._rstack.tif    # float32 ndarray, rescaled + rolled avg. stack
```

### `process.py`
Read `..._rstack.tif` from `data_path` folder and perform:
1) **Predictions**   
Predict object localization using the embedded deep-learning model. 
2) **Labeled mask**  
Average probabilities over time and define labelled objects as contiguous 
regions where probabilities are > 0.5. 


```bash
# Parameters
- data_path        # str, path to data directory
- min_size         # int, minimum size in pixels for considering segmented objects

# Outputs
- ..._probs.tif    # float32 ndarray, probabilities of deep-learning predictions
- ..._mask.tif     # uint8 ndarray, labelled segmented objects
```

### `analyse.py`
Read `..._stack.tif` and `..._mask.tif` from `data_path` folder and perform:
1) **Spatial filtering**   
Kernel mean filtering ignoring pixels outside of segmented objects.
2) **Temporal filtering**  
Pixel level Asymmetric Least Square (ALS) baseline (Low frequency components) subtraction.

```bash
# Parameters 
- data_path             # str, path to data directory

# Outputs
- ..._stack_filt.tif    # float32 ndarray, probabilities of deep-learning predictions
- ..._stack_bsub.tif    # uint8 ndarray, labelled segmented objects

# Note : filter parameters are fixed for the moment
```

## Comments