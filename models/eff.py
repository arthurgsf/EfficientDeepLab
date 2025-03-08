import tensorflow as tf

def se_block(x):
    n_channels = x.shape[-1]
    pooling = tf.keras.layers.GlobalAveragePooling2D()(x)
    dense1 = tf.keras.layers.Dense(n_channels, activation="relu")(pooling)
    dense2 = tf.keras.layers.Dense(n_channels, activation="sigmoid")(dense1)
    return dense2

def mbconv