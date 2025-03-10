import tensorflow as tf

def relu_bn_dropout(x, name=""):
    relu = tf.keras.layers.Activation("relu", name=f"{name}_relu")(x)
    bn = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(relu)
    dr = tf.keras.layers.Dropout(rate=0.2, name=f"{name}_dropout")(bn)
    return dr

def conv_block(x, channels, name=""):
    c1_name = f"{name}_conv1"
    c1 = tf.keras.layers.Conv2D(
        channels, (3, 3), 
        padding="same", 
        name=c1_name, 
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        kernel_initializer=initializer,
        bias_initializer=initializer
        )(x)
    c1 = relu_bn_dropout(c1, c1_name)

    c2_name = f"{name}_conv2"
    c2 = tf.keras.layers.Conv2D(
        channels, (3, 3), 
        padding="same", 
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        kernel_initializer=initializer,
        bias_initializer=initializer
        )(c1)
    c2 = relu_bn_dropout(c2, c2_name)

    return c2

def enc_block(x, channels, name=""):
    conv = conv_block(x, channels, name)
    downsampled = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv)
    return conv, downsampled
    
def dec_block(x, skip, channels, name=""):
    upsampled = tf.keras.layers.UpSampling2D(interpolation="bilinear")(x)
    concatenated = tf.keras.layers.Concatenate()([upsampled, skip])
    conv = conv_block(concatenated, channels, name)
    return conv

def Unet(input_shape, n_classes=1, activation = 'sigmoid'):
    weight_decay=1e-5
    global regularizer
    global initializer
    
    regularizer = tf.keras.regularizers.l2(weight_decay)
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    inputs = tf.keras.layers.Input(shape=(input_shape), name = "input")

    skip_1, enc_1  =  enc_block(inputs, 64, name="encoder1")
    skip_2, enc_2  =  enc_block(enc_1, 128, name="encoder2")
    skip_3, enc_3  =  enc_block(enc_2, 256, name="encoder3")
    skip_4, enc_4  =  enc_block(enc_3, 512, name="encoder4")

    bottleneck, _ = enc_block(enc_4, 512, name="bottleneck")

    dec_1 = dec_block(bottleneck, skip_4, 512, name="decoder1")
    dec_2 = dec_block(dec_1, skip_3, 256, name="decoder2")
    dec_3 = dec_block(dec_2, skip_2, 32, name="decoder3")
    dec_4 = dec_block(dec_3, skip_1, 64, name="decoder4")

    final_conv = tf.keras.layers.Conv2D(n_classes, kernel_size=(1, 1), activation="sigmoid")(dec_4)

    model = tf.keras.Model(inputs, final_conv, name="Unet")
    return model