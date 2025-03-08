import config as cfg
from test import test
from train import train
from utils import kfold
from models.efficientdeeplab import Backbones

if __name__ == "__main__":
    for backbone in [Backbones.B0, Backbones.B1, Backbones.B2]:
        for i, (train_patients, val_patients, test_patients) in enumerate(kfold(cfg.DATASET_PATH)):
            model, checkpoint_path = train(train_patients, val_patients, backbone, i = i)
            test(model, test_patients, checkpoint_path, i = i)