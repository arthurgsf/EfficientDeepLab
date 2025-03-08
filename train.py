import os
import utils
import config as cfg
import tensorflow as tf
import segmentation_models as sm
from models.efficientdeeplab import EfficientDeeplab
from models.sunet import SUnet
from models.unet import Unet
from generators import RandomSliceGenerator

def train(train_patients, val_patients, backbone, i = 0):
    # ===== DATASET CONFIGURATION ===== #
    train_dataset = tf.data.Dataset.from_generator(
        RandomSliceGenerator(train_patients, cfg.N_CLASSES),
        output_signature =
        (
            tf.TensorSpec(shape=cfg.IMAGE_SHAPE + (cfg.N_CHANNELS,)),
            tf.TensorSpec(shape=cfg.IMAGE_SHAPE + (cfg.N_CLASSES,))
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        RandomSliceGenerator(val_patients, cfg.N_CLASSES),
        output_signature =
        (
            tf.TensorSpec(shape=cfg.IMAGE_SHAPE + (cfg.N_CHANNELS,)),
            tf.TensorSpec(shape=cfg.IMAGE_SHAPE + (cfg.N_CLASSES,))
        )
    )

    # ===== MODEL DEFINITION ===== #
    model:tf.keras.Model = EfficientDeeplab(
        input_shape = cfg.IMAGE_SHAPE + (cfg.N_CHANNELS,),
        backbone = backbone,
        n_classes = cfg.N_CLASSES,
        activation = 'sigmoid'
    )

    # model = SUnet(
    #     input_shape = cfg.IMAGE_SHAPE + (cfg.N_CHANNELS,),
    #     n_classes = cfg.N_CLASSES
    # )

    # model = Unet(
    #     input_shape = cfg.IMAGE_SHAPE + (cfg.N_CHANNELS,),
    #     n_classes = cfg.N_CLASSES,
    #     activation='sigmoid'
    # )

    output_folder = f"output/{model.name}_{i}"
    checkpoint_path = f"{output_folder}/checkpoint"
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{checkpoint_path}/{model.name}",
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True,
        save_freq='epoch',
        mode='min',
        verbose=cfg.VERBOSE
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.1,
        patience=15
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode='min',
        factor      =   .1,
        patience    =   20,
        cooldown    =   15,
        min_lr      =   1e-6,
        min_delta   =   0.1
    )

    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=9e-1, weight_decay= 5e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer,
        loss      =   sm.losses.dice_loss,
        metrics   =   [
            cfg.precision_score,
            cfg.recall_score,
            cfg.iou_score,
            cfg.f_score,
        ]
    )

    H = model.fit(
        train_dataset.batch(cfg.BATCH_SIZE),
        validation_data=val_dataset.batch(1),
        epochs=cfg.EPOCHS,
        callbacks=[checkpoint, lr_scheduler],
        verbose = cfg.VERBOSE
    )

    utils.save_history(H.history, output_folder)

    return model, checkpoint_path