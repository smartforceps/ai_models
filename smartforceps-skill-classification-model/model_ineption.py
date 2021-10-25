# coding: utf-8

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


def test_model_2D(dropout_rate=0.25, activation='relu', input_dim=100, classes=2):
    start_neurons = 512
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=input_dim, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(classes, activation='softmax'))
    return model


def lstm(_x, n_steps, n_input, n_classes):
    n_hidden = 32
    _x = tf.transpose(_x, [1, 0, 2])  # (128, ?, 9)
    print("# transpose shape: ", _x.shape)  # transpose shape:  (128, ?, 9)
    _x = tf.reshape(_x, [-1, n_input])  # (n_step*batch_size, n_input)
    print("# reshape shape: ", _x.shape)  # reshape shape:  (?, 9)
    _x = Dense(
        inputs=_x,
        units=n_hidden,
        activation=tf.nn.relu,
    )
    print("# relu shape: ", _x.shape)  # relu shape:  (?, 32)
    _x = tf.split(_x, n_steps, 0)  # n_steps * (batch_size, n_hidden)
    # spilt makes _x.type from array --> list for static_rnn()
    print("# list shape: ", np.array(_x).shape)  # list shape:  (128,)
    print("# list unit shape: ", np.array(_x)[0].shape)  # list unit shape:  (?, 32)

    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_1_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=0.5)
    print("# cell_1 shape: ", lstm_cell_1.state_size)  # cell_1 shape:  LSTMStateTuple(c=32, h=32)

    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=0.5)
    print("# cell_2 shape: ", lstm_cell_2.state_size)  # cell_2 shape:  LSTMStateTuple(c=32, h=32)

    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1_drop, lstm_cell_2_drop], state_is_tuple=True)
    print("# multi cells shape: ", lstm_cells.state_size)
    # multi cells shape:  (LSTMStateTuple(c=32, h=32), LSTMStateTuple(c=32, h=32))

    outputs, states = tf.nn.static_rnn(cell=lstm_cells, inputs=_x, dtype=tf.float32)
    print("# outputs & states shape: ", np.array(outputs).shape, np.array(states).shape)
    # outputs & states shape:  (128,) (2, 2)

    lstm_last_output = outputs[-1]  # N to 1
    print("# last output shape: ", lstm_last_output.shape)  # last output shape:  (?, 32)

    lstm_last_output = tf.layers.dense(
        inputs=lstm_last_output,
        units=n_hidden,
        activation=tf.nn.relu
    )
    print("# fully connected shape: ", lstm_last_output.shape)  # fully connected shape:  (?, 32)

    prediction = tf.layers.dense(
        inputs=lstm_last_output,
        units=n_classes,
        activation=tf.nn.softmax
    )
    print("# prediction shape: ", prediction.shape)  # prediction shape:  (?, 6)
    return prediction


def cnn(X, num_labels):
    # CNN
    # convolution layer 1
    conv1 = tf.layers.conv1d(
        inputs=X,
        filters=64,
        kernel_size=2,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print("### convolution layer 1 shape: ", conv1.shape, " ###")

    # pooling layer 1
    pool1 = tf.layers.max_pooling1d(
        inputs=conv1,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer 1 shape: ", pool1.shape, " ###")

    # convolution layer 2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=128,
        kernel_size=2,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print("### convolution layer 2 shape: ", conv2.shape, " ###")

    # pooling layer 2
    pool2 = tf.layers.max_pooling1d(
        inputs=conv2,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer 2 shape: ", pool2.shape, " ###")

    # convolution layer 3
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=256,
        kernel_size=2,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print("### convolution layer 3 shape: ", conv3.shape, " ###")

    # pooling layer 3
    pool3 = tf.layers.max_pooling1d(
        inputs=conv3,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer 3 shape: ", pool3.shape, " ###")

    # flat output
    l_op = pool3
    shape = l_op.get_shape().as_list()
    flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
    print("### flat shape: ", flat.shape, " ###")

    # fully connected layer 1
    fc1 = tf.layers.dense(
        inputs=flat,
        units=100,
        activation=tf.nn.tanh
    )
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
    print("### fully connected layer 1 shape: ", fc1.shape, " ###")
    # bn_fc1 = tf.layers.batch_normalization(fc1, training=training)
    # bn_fc1_act = tf.nn.relu(bn_fc1)

    # fully connected layer 1
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=100,
        activation=tf.nn.tanh
    )
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
    print("### fully connected layer 2 shape: ", fc2.shape, " ###")
    # bn_fc2 = tf.layers.batch_normalization(fc2, training=training)
    # bn_fc2_act = tf.nn.relu(bn_fc2)

    # fully connected layer 3
    fc3 = tf.layers.dense(
        inputs=fc2,
        units=num_labels,
        activation=tf.nn.softmax
    )
    print("### fully connected layer 3 shape: ", fc3.shape, " ###")

    # prediction
    # y_ = tf.layers.batch_normalization(fc3, training=training)
    y_ = fc3
    print("### prediction shape: ", y_.get_shape(), " ###")
    return y_


def cnnlstm(X, N_TIME_STEPS, N_CLASSES):
    N_HIDDEN_UNITS = 32
    conv1 = tf.layers.conv1d(inputs=X, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv1d(inputs=conv1, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    # conv3 = tf.layers.conv1d(inputs=conv2, filters=32, kernel_size=5, strides=1, padding='same', activation = tf.nn.relu)
    n_ch = 32
    lstm_in = tf.transpose(conv2, [1, 0, 2])  # reshape into (seq_len, batch, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*batch, n_channels)
    # To cells
    lstm_in = tf.layers.dense(lstm_in, N_HIDDEN_UNITS, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, N_TIME_STEPS, 0)

    # Add LSTM layers
    lstm = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    cell = tf.contrib.rnn.MultiRNNCell(lstm)
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32)

    # We only need the last output tensor to pass into a classifier
    pred = tf.layers.dense(outputs[-1], units=N_CLASSES, activation=tf.nn.softmax)
    return pred
