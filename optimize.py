import utils
import pickle
import optuna as tuna
import tensorflow as tf
from config import args
from train import train, test
import segthor_preprocessing as pre

def obj(trial:tuna.Trial):
    # hyper-parameters
    lower_bound = trial.suggest_int("lower_bound", -1000, 0, step=100)
    upper_bound = trial.suggest_int("upper_bound", 10, 100, step=10)

    preprocessing_pipeline = pre.Pipeline([
        pre.intensity_clip(lower_bound, upper_bound),
        pre.center_crop(args.input_shape),
        pre.mean_std_norm(),
        pre.expand_dims()
    ])

    model = train(preprocessing_pipeline, folder_name = str(trial.number))
    test_dice = test(model, preprocessing_pipeline, folder_name = str(trial.number))

    with open(f"{utils.get_model_path(model, trial)}params", "wb") as f:
        pickle.dump(trial.params, f)

    return test_dice

if __name__ == "__main__":
    # tf setup
    devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    study = tuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=args.opt_epochs)