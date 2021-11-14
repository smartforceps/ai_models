from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D, Activation, concatenate
from tensorflow.keras import backend as K


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_data_format() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (1, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (1, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def SF_unet(subseq=224, filters=32, dropout_val=0.2, batch_norm=True, INPUT_CHANNELS=3, OUTPUT_MASK_CHANNELS=6):
    if K.image_data_format() == 'th':
        inputs = Input((INPUT_CHANNELS, 1, subseq))
        axis = 1
    else:
        inputs = Input((1, subseq, INPUT_CHANNELS))
        axis = 3

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(1, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2 * filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(1, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4 * filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(1, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8 * filters, 0, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(1, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16 * filters, 0, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(1, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32 * filters, 0, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(1, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16 * filters, 0, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(1, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8 * filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(1, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4 * filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(1, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2 * filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(1, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="SF_unet")
    return model

