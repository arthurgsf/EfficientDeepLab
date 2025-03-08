import os
import cv2 as cv
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


SOURCE_DATASET = f"{os.path.expanduser('~')}/Datasets/SEGTHOR"

IMAGE_SIZE = (304, 304)
DST_FOLDER = f"datasets/noclip_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"

def center_crop(image, mask, shape):
    center = image.shape
    w = shape[0]
    h = shape[1]
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    crop_img = image[int(y):int(y+h), int(x):int(x+w)]
    crop_label = mask[int(y):int(y+h), int(x):int(x+w)]
    return crop_img, crop_label

def extract_patient(p, pid):
    #lendo volume
    volume = nib.load(f"{p}/{pid}.nii.gz").get_fdata()
    volume = np.clip(volume, -1000, 3000) # intensisty windowing
    volume = ((volume + 1000)/4000)*255 #*norm 0-255
    volume = volume.astype(np.uint8)
    
    # lendo GT
    gt = nib.load(f"{p}/GT.nii.gz").get_fdata()
    gt = gt.astype(np.uint8)

    for i in range(volume.shape[-1]):
        x = volume[:,:,i]
        y = gt[:,:,i]
        y[y!=3] = 0
        y[y==3] = 1

        x, y = center_crop(x, y, IMAGE_SIZE)
        # x = cv.resize(x, IMAGE_SIZE)
        # y = cv.resize(y, IMAGE_SIZE, interpolation = cv.INTER_NEAREST)
        
        yield x, y

if __name__ == "__main__":
    os.makedirs(DST_FOLDER)
    
    patients = glob(f"{SOURCE_DATASET}/*")
    # print(patients)
    
    for p in tqdm(patients):
        pid = Path(p).stem #Patient_xx
        os.makedirs(f"{DST_FOLDER}/{pid}/")
        
        for i, (img, label) in enumerate(extract_patient(p, pid)):
            cv.imwrite(f"{DST_FOLDER}/{pid}/{pid}_{i}_im.png", img)
            cv.imwrite(f"{DST_FOLDER}/{pid}/{pid}_{i}_gt.png", label)