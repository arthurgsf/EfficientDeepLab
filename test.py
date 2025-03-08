import utils
import numpy as np
import config as cfg
import tensorflow as tf
from generators import PatientWiseGenerator
from postprocessing import biggest_3D_object

def test(model, test_patients, checkpoint_path, i = 0):
    # ===== TEST ===== #
    model.load_weights(f"{checkpoint_path}/{model.name}")

    metrics = {
        "dice":[],
        "precision":[],
        "recall":[],
        "iou_score":[],
    }

    metrics_post_processed = {
        "dice":[],
        "precision":[],
        "recall":[],
        "iou_score":[],
    }

    for patient in test_patients:
        generator = PatientWiseGenerator(patient, cfg.N_CLASSES)
        patient_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=
            (
                tf.TensorSpec(shape=cfg.IMAGE_SHAPE + (cfg.N_CHANNELS,)),
                tf.TensorSpec(shape=cfg.IMAGE_SHAPE + (cfg.N_CLASSES,))
            )
        )
        
        volume_true = []
        for (x, y) in patient_dataset:
            volume_true.append(y)
        volume_true = np.squeeze(np.array(volume_true))

        volume_pred = model.predict(patient_dataset.batch(5), verbose=0)
        volume_pred = np.round(volume_pred)
        volume_pred = np.squeeze(volume_pred)

        metrics["dice"].append(cfg.f_score(volume_true, volume_pred))
        metrics["precision"].append(cfg.precision_score(volume_true, volume_pred))
        metrics["recall"].append(cfg.recall_score(volume_true, volume_pred))
        metrics["iou_score"].append(cfg.iou_score(volume_true, volume_pred))

        volume_post_processed = biggest_3D_object(volume_pred)
    
        metrics_post_processed["dice"].append(cfg.f_score(volume_true, volume_post_processed))
        metrics_post_processed["precision"].append(cfg.precision_score(volume_true, volume_post_processed))
        metrics_post_processed["recall"].append(cfg.recall_score(volume_true, volume_post_processed))
        metrics_post_processed["iou_score"].append(cfg.iou_score(volume_true, volume_post_processed))
    
    output_folder = f"output/{model.name}_{i}"
    utils.store_test_metrics(metrics, output_folder, "test_metrics")
    utils.store_test_metrics(metrics_post_processed, output_folder, "test_metrics_post_processing")