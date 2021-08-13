# encoding=utf8
import numpy as np
from numpy import mean
from numpy import std
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
# from keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import pickle
# import unet_224_model
import models_task
import data_load_task
import model_info
import common
# from smartforceps_dl_prediction_models_tf2.smartforceps_task_model import models_task
# from smartforceps_dl_prediction_models_tf2.smartforceps_task_model import data_load_task
# from smartforceps_dl_prediction_models_tf2.smartforceps_task_model import model_info
# from smartforceps_dl_prediction_models_tf2.smartforceps_task_model import common
import pandas as pd
import logging
import argparse
import time
import os
import gc

import warnings

warnings.filterwarnings('ignore')

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/graphs'):
    os.makedirs('results/graphs')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, nargs='?', default='Smartforceps', help="dataset name")
parser.add_argument('--subseq', type=int, nargs='?', default=224, help="loss name")
# different network length (block number)
parser.add_argument('--block', type=str, nargs='?', default='5', help="block number")
parser.add_argument('--net', type=str, nargs='?', default='unet', help="net name")

args = parser.parse_args()
subseq = args.subseq
# file_log = 'UNET_'+args.block+'_'+args.dataset+'_'+str(subseq)+'.log'
# file_log = 'MASK_'+args.dataset+'_'+str(subseq)+'.log'
file_log = './results/' + args.net + '_' + args.dataset + '_' + str(subseq) + '.log'

logging.basicConfig(filename=file_log, level=logging.DEBUG)
model_info.begin()

read_processed_data = data_load_task.load_data(args.dataset, subseq=subseq)

X = read_processed_data[0]
y = read_processed_data[1]
X_train = read_processed_data[2]
X_test = read_processed_data[3]
y_train = read_processed_data[4]
y_test = read_processed_data[5]
y_map_train = read_processed_data[6]
wtable_train = read_processed_data[7]
N_FEATURES = read_processed_data[8]
y_map = read_processed_data[9]
act_classes = read_processed_data[10]
class_names = read_processed_data[11]


# user defined loss function
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true, y_pred):
    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.math.log(yc), axis=0) / wtable_train))
    return loss


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

clfs = []
oof_preds = np.zeros((y_train.shape[0], y_train.shape[1]))
epochs = 250
batch_size = 32
optim_type = 'adam'
learning_rate = 0.001
sum_time = 0
for fold_, (trn_, val_) in enumerate(folds.split(np.zeros(X_train.shape[0]), np.zeros(X_train.shape[0]))):

    trainX, trainy = X_train[trn_, :, :], y_train[trn_, :]
    validX, validy = X_train[val_, :, :], y_train[val_, :]

    if args.net == 'lstm_model':
        model = models_task.lstm_model(timesteps_count=subseq,
                                       feature_count=N_FEATURES,
                                       activation='relu',
                                       n_output=act_classes)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    # callbacks_list = [
    #     ModelCheckpoint("./results/keras_task.model",
    #                     monitor='val_loss',
    #                     mode='min',
    #                     save_best_only=True,
    #                     save_weights_only=True,
    #                     verbose=0),
    #     EarlyStopping(monitor='val_loss', patience=1)
    # ]
    callbacks_list = [
        ModelCheckpoint("./results/keras_task.model",
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=0)
    ]

    print("processing fold: ", fold_ + 1)
    logging.info('mean_time={}'.format(str(fold_ + 1)))

    history = model.fit(trainX, trainy,
                        validation_data=(validX, validy),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        # verbose=0,
                        callbacks=callbacks_list)

    acc = np.array(history.history['accuracy'])
    loss = np.array(history.history['loss'])
    val_acc = np.array(history.history['val_accuracy'])
    val_loss = np.array(history.history['val_loss'])
    for i in range(acc.shape[0]):
        logging.info(
            'Epoch: {} loss: {:.4f} accuracy{:.4f} val_loss: {:.4f} val_accuracy{:.4f}%\n'.format(i + 1, loss[i],
                                                                                                  acc[i],
                                                                                                  val_loss[i],
                                                                                                  val_acc[i]))

    print('Loading Best Model')
    model.load_weights('./results/keras_task.model')
    # serialize model to JSON
    model_json = model.to_json()
    with open("./results/task_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./results/task_model.h5")
    print("Saved model to disk")
    # get predicted probabilities for each class
    valid_pred = model.predict(validX, batch_size=batch_size)
    oof_preds[val_, :] = valid_pred
    print('MULTI WEIGHTED LOG LOSS : %.5f '
          % common.multi_weighted_logloss(validy, valid_pred))

    clfs.append(model)

# print model summary
print('model summary:')
print(str(model.summary()))

logging.info("model summary:")
model.summary(print_fn=logging.info)

# take the class with the highest probability from the train predictions
print('classification report:\n', classification_report(pd.DataFrame(y_map),
                                                        np.argmax(oof_preds, axis=1)))
print('MULTI WEIGHTED LOG LOSS : %.5f '
      % common.multi_weighted_logloss(y_train, oof_preds))

print('__________________________________________________________________________________________________')
print('testing the model and showing results:')

start = time.perf_counter()
model.evaluate(X_test, y_test, batch_size=batch_size)
end = time.perf_counter()
sum_time += (end - start)
print('time passed for testing: ', str(end - start), 'sec')
logging.info('mean_time={}'.format(str(end - start)))
y_test_argmax = np.argmax(y_test, axis=1)
labels_test_unary = y_test_argmax.reshape(y_test_argmax.size)
file_labels_test_unary = './results/' + 'labels_gd_' + args.dataset + '_' + \
                         str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_labels_test_unary, labels_test_unary)

y_pred_prob = model.predict(X_test, batch_size=batch_size)
y_pred = np.argmax(y_pred_prob, axis=1)
print('prediction results data shape: ', y_pred_prob.shape)
file_y_pred = './results/' + 'y_pred_' + args.dataset + '_' + \
              str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_y_pred, y_pred)
file_y_pred_prob = './results/' + 'y_pred_prob_' + args.dataset + '_' + \
                   str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_y_pred_prob, y_pred_prob)

label_index = list(range(1, act_classes + 1))
accuracy, precision, recall, fscore, fw = common.checkAccuracy(labels_test_unary + 1, y_pred + 1, label_index)
print("testing confusion matrix:")
print(common.createConfusionMatrix(labels_test_unary + 1, y_pred + 1, label_index))

logging.info("testing confusionmatrix:")
logging.info('testing confusionmatrix:{}'.format(common.createConfusionMatrix(labels_test_unary + 1,
                                                                              y_pred + 1,
                                                                              label_index)))

print('testing acc:{}'.format(accuracy))
logging.info('testing acc:{}'.format(accuracy))
print('testing fscore:{}'.format(fscore))
logging.info('testing fscore:{}'.format(fscore))
print('testing weighted fscore:{}'.format(fw))
logging.info('testing weighted fscore:{}'.format(fw))

# plot results
print("saving the confusion matrix graph")
common.plot_confusion_matrix(common.createConfusionMatrix(labels_test_unary + 1, y_pred + 1, label_index),
                             plot_classes=class_names,
                             normalize=True,
                             title='Confusion matrix')
print("saving the history result graphs")
common.plot_loss_acc(history)

print("results saved in _results_ folder")
