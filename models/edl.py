import tensorflow as tf

def atrous_conv(n_filters, kernel_size=(3, 3), dilation_rate=1, name=None):
    # name = name_prefix + "_atrous_conv" if name_prefix else "atrous_conv"
    conv = tf.keras.layers.Conv2D(
        n_filters, 
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        name=name
        )
    return conv

def aspp(x):
    conv_list = []
    for r in [24, 18, 12, 6, 1]:
        conv_list.append(atrous_conv(128, dilation_rate=r, name=f"aspp_conv_r{r}")(x))
    c = tf.keras.layers.Concatenate(name="aspp_concatenation")(conv_list)
    return c

def encoder(x):
    c1 = atrous_conv(32, dilation_rate=4)(x)
    t = aspp(c1)
    c2 = atrous_conv(64, kernel_size=(1, 1), dilation_rate=1)(t)
    upsampled = tf.keras.layers.UpSampling2D(size=(4, 4))(c2)
    return upsampled

def decoder(x, encoder_output):
    c1 = atrous_conv(32, dilation_rate=4)(x)
    c2 = atrous_conv(64, kernel_size=(1, 1), dilation_rate=1)(c1)
    c = tf.keras.layers.Concatenate()([encoder_output, c2])
    c3 = atrous_conv(64, dilation_rate=1)(c)
    c4 = atrous_conv(64, kernel_size=(1, 1), dilation_rate=1)(c3)
    upsampled = tf.keras.layers.UpSampling2D(size=(4, 4))(c4)
    return upsampled

if __name__ == "__main__":
    input = tf.keras.layers.Input((256, 256, 3))

    encoder_output = encoder(input)
    output = decoder(input, encoder_output)

    model = tf.keras.Model(input, output)
    model.summary()