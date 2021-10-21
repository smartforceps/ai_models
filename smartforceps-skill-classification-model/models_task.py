# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, LSTM, Embedding, Lambda
from tensorflow.keras.layers import SpatialDropout2D, Activation, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras import backend as K


# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)cdcd
# INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)pyth
# OUTPUT_MASK_CHANNELS = 6

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


def lstm_model(units_size=100, timesteps_count=200, feature_count=2, activation='relu', n_output=2):
    model = Sequential()
    model.add(LSTM(units_size, input_shape=(timesteps_count, feature_count)))
    model.add(Dropout(0.5))
    model.add(Dense(units_size, activation=activation))
    model.add(Dense(n_output, activation='softmax'))
    return model
