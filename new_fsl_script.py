#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os 
import glob
import os
from multiprocessing import Pool
import functools
os.getcwd()


# In[7]:


def resize_3d_tensor(volume, target_shape=(112, 112, 112)):
    vol_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    resized = F.interpolate(vol_tensor, size=target_shape, mode='trilinear', align_corners=False)
    return resized.squeeze(0).squeeze(0) 


def intensity_normalization(volume: np.array, clip_ratio: float = 99.5): 
    assert np.min(volume) == 0.0, "Input volume must have minimum intensity 0"
    volume_max = np.percentile(volume, clip_ratio)
    volume = np.clip(volume / volume_max, 0, 1)
    return volume


# In[8]:


def run_fsl_processing(image_path: Path, ref: Path):
    # reorient images to a standard orientation - Right-Anterior-Superior (RAS)
    # Right (R) – X-axis: The direction from the patient's left side to their right side.
    # Anterior (A) – Y-axis: The direction from the patient's back to their front.
    # Superior (S) – Z-axis: The direction from the patient's feet to their head.
    fslreorient2std_path = image_path.replace(".nii.gz", "_fslreorient2std.nii")
    os.system(f'fslreorient2std {image_path} {fslreorient2std_path}')


    # compute a more accurate field of view (FOV) for brain images
    # robustfov tries to crop out extra space around the brain to focus only on the relevant region.
    robust_path = fslreorient2std_path.replace("_fslreorient2std.nii", "_robust.nii")
    os.system(f'robustfov -i {fslreorient2std_path} -r {robust_path}')


    # perform skull stripping on MRI images
    # print(f'bet {preprocessed_image_path} {preprocessed_image_path} -R')
    bet_path = robust_path.replace("_robust.nii", "_bet.nii") #TODO: Error: input image /data/timeleap-shared/ADNI_1_2_GO/I374298_robust not valid
    # print(f'bet {robust_path} {bet_path} -R')
    # first pass
    os.system(f'bet2 {robust_path} {bet_path} -f 0.5')
    # second pass for refinement
    os.system(f'bet2 {bet_path} {bet_path} -f 0.5')    
    # linear registration of brain images. It aligns (registers) one image to another.
    # FLIRT performs linear registration, meaning it uses affine transformations to align one image (the "input") to another (the "reference").
    # rotation, scaling, Translation, shearing 
    flirt_path = bet_path.replace("_bet.nii", "_flirt.nii")
    os.system(f'flirt -in {bet_path} -ref {ref} -out {flirt_path}')


    # performs segmentation of brain images. It segments the brain into different tissue types, typically including gray matter, white matter, and cerebrospinal fluid (CSF).
    # "fast" saves output file as {file_name}_restore.nii.gz
    # also bias-field correction --> this is output 
    fast_path = flirt_path.replace("_flirt.nii", "_fast.nii")
    os.system(f'fast --nopve -B -o {fast_path} {flirt_path} ') # TODO # Files end with fast_restore.nii.gz 
    # preprocessed_image_path_fsl = Path(str(preprocessed_image_path).replace(".nii.gz", "_restore.nii"))
    # return preprocessed_image_path_fsl


# In[9]:


os.chdir("/data/cmn_vamal/mwilson/new_data/new_data_folder/nii_folder")
os.getcwd()


# In[ ]:


def process_file(f, ref):
    fast_restore = f.replace(".nii.gz", "_fast_restore.nii.gz")
    if os.path.exists(fast_restore):
        print(f"Skipping {f}, already processed")
        return
    run_fsl_processing(f, ref)

ref = "/home/AD/lsucipto/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
files = sorted(glob.glob("*.nii.gz"))

with Pool(processes=30) as pool:
    pool.map(functools.partial(process_file, ref=ref), files)


# In[ ]:




