import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Conv1D, Conv2D, Conv2DTranspose, MaxPool1D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import LSTM, Embedding, Lambda
from tensorflow.keras.layers import SpatialDropout2D, Activation, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras import backend as K
from tensorflow import keras


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


def build_inception_model(input_shape, nb_classes, depth):
    input_layer = Input(input_shape)

    x = input_layer
    input_res = input_layer

    def inception_module(input_tensor, stride=1, activation='linear',
                         kernel_size=41, nb_filters=32):
        input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                    strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                        padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    for d in range(depth):

        x = inception_module(x)

        if d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)

    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
