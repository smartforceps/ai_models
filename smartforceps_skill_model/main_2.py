# encoding=utf8
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
# from keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import pickle
# import unet_224_model
import models_skill
import data_load_skill
import model_info
import common
# from smartforceps_dl_prediction_models_tf2.smartforceps_skill_model import models_skill
# from smartforceps_dl_prediction_models_tf2.smartforceps_skill_model import data_load_skill
# from smartforceps_dl_prediction_models_tf2.smartforceps_skill_model import model_info
# from smartforceps_dl_prediction_models_tf2.smartforceps_skill_model import common
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

read_processed_data = data_load_skill.load_data(args.dataset)

train_x = read_processed_data[0]
test_x = read_processed_data[1]
train_y = read_processed_data[2]
test_y = read_processed_data[3]
y_categorical_train = read_processed_data[4]
y_map = read_processed_data[5]
wtable = read_processed_data[6]
y_categorical_test = read_processed_data[7]
act_classes = read_processed_data[8]
class_names = read_processed_data[9]


classes = np.unique(train_y).tolist()
full_train = train_x.copy()
train_features = full_train[['LeftCalibratedForceValue', 'RightCalibratedForceValue']]

train_mean = train_features.mean(axis=0)
# full_train.fillna(train_mean, inplace=True)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Standard Scaling the input (imp.)

full_train_new = train_features.copy()
ss = StandardScaler()
full_train_ss = ss.fit_transform(full_train_new)


# user defined loss function
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true, y_pred):
    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.math.log(yc), axis=0) / wtable))
    return loss


# defining simple model in keras

K.clear_session()

clfs = []
oof_preds = np.zeros((len(full_train_ss), len(classes)))
epochs = 2000
batch_size = 32
optim_type = 'adam'
learning_rate = 0.001
sum_time = 0
for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):

    trainX, trainy = full_train_ss[trn_], y_categorical_train[trn_]
    validX, validy = full_train_ss[val_], y_categorical_train[val_]

    if args.net == 'test_model_2D':
        model = models_skill.test_model_2D(dropout_rate=0.25, activation='relu',
                                           input_dim=full_train_ss.shape[1], classes=len(classes))
    elif args.net == 'inception_time':
        model = models_skill.Classifier_INCEPTION(
            # output_directory='./results/',
            input_shape=trainX.shape[1:],
            nb_classes=len(classes),
            verbose=True,
            build=True)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)

    model.compile(loss=mywloss, optimizer=optim, metrics=['accuracy'])

    # callbacks_list = [
    #     ModelCheckpoint("./results/keras_skill.model",
    #                     monitor='val_loss',
    #                     mode='min',
    #                     save_best_only=True,
    #                     save_weights_only=True,
    #                     verbose=1),
    #     EarlyStopping(monitor='accuracy', patience=1)
    # ]
    callbacks_list = [
        ModelCheckpoint("./results/keras_skill.model",
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1)
    ]

    print("processing fold: ", fold_ + 1)
    logging.info('mean_time={}'.format(str(fold_ + 1)))

    history = model.fit(trainX, trainy,
                        validation_data=(validX, validy),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1,
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
    model.load_weights('./results/keras_skill.model')
    # serialize model to JSON
    model_json = model.to_json()
    with open("./results/skill_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./results/skill_model.h5")
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
logging.info('model summary:{}'.format(model.summary()))


# take the class with the highest probability from the train predictions
print('classification report:\n', classification_report(pd.DataFrame(y_map),
                                                        np.argmax(oof_preds, axis=1)))
print('MULTI WEIGHTED LOG LOSS : %.5f '
      % common.multi_weighted_logloss(y_categorical_train, oof_preds))

print('__________________________________________________________________________________________________')
print('testing the model and showing results:')

full_test = test_x.copy()
test_features = full_test[['LeftCalibratedForceValue', 'RightCalibratedForceValue']]
test_features[train_features.columns] = test_features[train_features.columns].fillna(train_mean)
full_test_ss = ss.transform(test_features[train_features.columns])


start = time.perf_counter()
model.evaluate(full_test_ss, y_categorical_test, batch_size=batch_size)
end = time.perf_counter()
sum_time += (end - start)
print('time passed for testing: ', str(end - start), 'sec')
logging.info('mean_time={}'.format(str(end - start)))
file_labels_test_unary = './results/' + 'labels_gd_' + args.dataset + '_' + \
                         str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_labels_test_unary, test_y)

y_pred_prob = model.predict(full_test_ss, batch_size=batch_size)
y_pred = np.argmax(y_pred_prob, axis=1)
print('prediction results data shape: ', y_pred_prob.shape)
file_y_pred = './results/' + 'y_pred_' + args.dataset + '_' + \
              str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_y_pred, y_pred)
file_y_pred_prob = './results/' + 'y_pred_prob_' + args.dataset + '_' + \
                   str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_y_pred_prob, y_pred_prob)

label_index = list(range(1, act_classes + 1))
accuracy, precision, recall, fscore, fw = common.checkAccuracy(test_y + 1, y_pred + 1, label_index)
print("testing confusion matrix:")
print(common.createConfusionMatrix(test_y + 1, y_pred + 1, label_index))

logging.info("testing confusionmatrix:")
logging.info('testing confusionmatrix:{}'.format(common.createConfusionMatrix(test_y + 1,
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
common.plot_confusion_matrix(common.createConfusionMatrix(test_y + 1, y_pred + 1, label_index),
                             plot_classes=class_names,
                             normalize=True,
                             title='Confusion matrix')
print("saving the history result graphs")
common.plot_loss_acc(history)

print("results saved in _results_ folder")
