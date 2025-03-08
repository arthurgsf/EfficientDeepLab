import numpy as np
from enum import Enum
import tensorflow as tf
import tensorflow_advanced_segmentation_models as tasm

class Backbones(Enum):
    B0 = 0
    B1 = 1
    B2 = 2
    B3 = 3
    B4 = 4
    B5 = 5
    B6 = 6
    B7 = 7

def EfficientDeeplab(input_shape:tuple, backbone:Backbones, n_classes = 1, activation = "sigmoid") -> tf.keras.Model:
    """
        Instantiates efficient deeplab model with the passed backbone and input_shape
    """
    efficient, layers, _ = tasm.create_base_model(
        name    =   f"efficientnetb{backbone.value}",
        height  =   input_shape[1], 
        width   =   input_shape[0],
        pooling="max_pooling"
    )
    
    deeplab = tasm.DeepLabV3plus(
        n_classes           =   n_classes,
        base_model          =   efficient,
        output_layers       =   layers,
        backbone_trainable  =   True,
        final_activation    =   activation)
    
    if input_shape[2] != 3:
        # map N channels data to 3 channels
        inputs = tf.keras.Input(shape=input_shape)
        conv = tf.keras.layers.Conv2D(
            3, (1, 1), 
            padding="same", 
            name='channel_expansion',
            trainable = True)(inputs)
        outputs = deeplab(conv)
        return tf.keras.Model(inputs, outputs, name=f"EfficientDeeplab{backbone.name}")
    else:
        return tf.keras.Model(deeplab.inputs, deeplab.outputs, name=f"EfficientDeeplab{backbone.name}")