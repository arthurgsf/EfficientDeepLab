import pickle
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.model_selection import KFold

def save_history(history, model_path):
    with open(f"{model_path}/history", "wb") as f:
            pickle.dump(history, f)

def store_test_metrics(metrics, model_path, filename):
    with open(f"{model_path}/{filename}", "wb") as f:
        pickle.dump(metrics, f)

def get_patients(folder):
    files = glob(f"{folder}/*_im.png")
    splitted_paths = [Path(p).stem.split("_") for p in files]
    unique_patients = set([f"{p[0]}_{p[1]}" for p in splitted_paths])
    return unique_patients

def kfold(dataset_path):
    patients = glob(f"{dataset_path}/*")
    patients = np.array(patients)

    rkf = KFold(n_splits = 3, random_state = 42, shuffle = True)

    for train, base_test in rkf.split(patients):
        train_patients = patients[train]
        center = len(base_test)//2
        
        val_idx = base_test[:center]
        val_patients = patients[val_idx]

        test_idx = base_test[center:]
        test_patients = patients[test_idx]

        yield train_patients, val_patients, test_patients