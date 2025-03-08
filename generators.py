import random
import cv2 as cv
import numpy as np
from glob import glob
import tensorflow as tf
from pathlib import Path
from preprocessing import mean_std_norm

class RandomSliceGenerator:
    """
        Generate random slices.
    """
    def __init__(self, patients, n_classes, shuffle = True, norm=True):
        
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.norm = norm
        
        self.x_paths = []
        
        for patient in patients:
            self.x_paths += glob(f"{patient}/*_im.png")
        self.y_paths = [p.replace("im", "gt") for p in self.x_paths]
    
    def __call__(self):
        data = list(zip(self.x_paths, self.y_paths))
        
        if self.shuffle:
            random.shuffle(data)

        for x_path, y_path in data:
            y = cv.imread(y_path, cv.IMREAD_UNCHANGED)
            trachea_present = np.count_nonzero(y) > 0
            if trachea_present:
                y = np.expand_dims(y, axis=-1)
                x = cv.imread(x_path, cv.IMREAD_UNCHANGED)
                if self.norm:
                    x = (x - np.mean(x))/np.std(x)
                x = np.expand_dims(x, axis = -1)
                yield x, y

class PatientWiseGenerator:
    """
        Generate slices in order per-patient.
    """
    def __init__(self, patient, n_classes, norm=True):
        self.x_paths = glob(f"{patient}/*_im.png")
        self.x_paths = sorted(self.x_paths, key=lambda p: int(Path(p).stem.split("_")[-2]))
        self.y_paths = [p.replace("im", "gt") for p in self.x_paths]
        self.n_classes = n_classes
        self.norm = norm
    
    def __call__(self):
        data = zip(self.x_paths, self.y_paths)
        for x_path, y_path in data:
            x = cv.imread(x_path, cv.IMREAD_UNCHANGED)
            if self.norm:
                x = (x - np.mean(x))/np.std(x)
            
            y = cv.imread(y_path, cv.IMREAD_UNCHANGED)
            y = np.expand_dims(y, axis=-1)
            x = np.expand_dims(x, axis = -1)
            
            
            yield x, y