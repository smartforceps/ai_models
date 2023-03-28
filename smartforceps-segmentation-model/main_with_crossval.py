# encoding=utf8
import numpy as np
from numpy import mean, std
from sklearn.model_selection import StratifiedKFold
# from keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# import unet_224_model
import unet_model
import data_load_segment
import unet_info
import common
import pandas as pd
import logging
import argparse
import time
import os

import warnings

warnings.filterwarnings('ignore')

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/graphs'):
    os.makedirs('results/graphs')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, nargs='?', default='WISDMar', help="dataset name")
parser.add_argument('--subseq', type=int, nargs='?', default=224, help="loss name")
# different network length (block number)
parser.add_argument('--block', type=str, nargs='?', default='5', help="block number")
parser.add_argument('--net', type=str, nargs='?', default='unet', help="net name")

args = parser.parse_args()
subseq = args.subseq
# file_log = 'UNET_'+args.block+'_'+args.dataset+'_'+str(subseq)+'0501.log'
# file_log = 'MASK_'+args.dataset+'_'+str(subseq)+'0501.log'
file_log = './results/' + args.net + '_' + args.dataset + '_' + str(subseq) + '.log'

logging.basicConfig(filename=file_log, level=logging.DEBUG)
unet_info.begin()

start_total = time.perf_counter()

read_processed_data = data_load_segment.load_data(args.dataset, subseq=subseq)

X = read_processed_data[0]
y = read_processed_data[1]
X_train = read_processed_data[2]
X_val = read_processed_data[3]
X_test = read_processed_data[4]
y_train = read_processed_data[5]
y_val = read_processed_data[6]
y_test = read_processed_data[7]
N_FEATURES = read_processed_data[8]
y_map_train = read_processed_data[9]
y_map_val = read_processed_data[10]
y_map_test = read_processed_data[11]
act_classes = read_processed_data[12]
class_names = read_processed_data[13]


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


clfs = []
oof_preds = np.zeros((y_train.shape[0], y_train.shape[2], y_train.shape[3]))
epochs = 50
batch_size = 128
optim_type = 'adam'
learning_rate = 0.001
sum_time = 0


folds_train = []
trn_train = []
val_train = []
for fold_, (trn_, val_) in enumerate(folds.split(np.zeros(X_train.shape[0]), np.zeros(X_train.shape[0]))):
    folds_train.append(fold_)
    trn_train.append(trn_)
    val_train.append(val_)

folds_val = []
trn_val = []
val_val = []
for fold_, (trn_, val_) in enumerate(folds.split(np.zeros(X_val.shape[0]), np.zeros(X_val.shape[0]))):
    folds_val.append(fold_)
    trn_val.append(trn_)
    val_val.append(val_)

folds_test = []
trn_test = []
val_test = []
for fold_, (trn_, val_) in enumerate(folds.split(np.zeros(X_test.shape[0]), np.zeros(X_test.shape[0]))):
    folds_test.append(fold_)
    trn_test.append(trn_)
    val_test.append(val_)

accuracy_folds = []
precision_folds = []
recall_folds = []
fscore_folds = []
fw_folds = []


for fold_ in folds_train:

    trn_ = trn_train[fold_]
    val_ = val_val[fold_]
    test_ = val_test[fold_]

    trainX, trainy = X_train[trn_, :, :, :], y_train[trn_, :, :, :]
    validX, validy = X_val[val_, :, :, :], y_val[val_, :, :, :]
    testX, testy = X_test[test_, :, :, :], y_test[test_, :, :, :]

    if (args.net == 'unet') and (args.block == '5'):
        model = unet_model.ZF_UNET_224(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                       OUTPUT_MASK_CHANNELS=act_classes)
    elif (args.net == 'unet') and (args.block == '4'):
        model = unet_model.ZF_UNET_224_4(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                         OUTPUT_MASK_CHANNELS=act_classes)
    elif (args.net == 'unet') and (args.block == '3'):
        model = unet_model.ZF_UNET_224_3(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                         OUTPUT_MASK_CHANNELS=act_classes)
    elif (args.net == 'unet') and (args.block == '2'):
        model = unet_model.ZF_UNET_224_2(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                         OUTPUT_MASK_CHANNELS=act_classes)
    elif args.net == 'fcn':
        model = unet_model.FCN(inputsize=512, deconv_output_size=512, INPUT_CHANNELS=N_FEATURES,
                               num_classes=act_classes, filters=32)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss',
    #                                factor=np.sqrt(0.1),
    #                                cooldown=0,
    #                                patience=10, min_lr=1e-12)
    # callbacks = [lr_reducer]

    # early_stopper = EarlyStopping(monitor='val_loss',
    #                             patience=30)
    #
    # callbacks = [lr_reducer, early_stopper]

    # callbacks_list = [
    #     ModelCheckpoint("./results/keras_test.model",
    #                     monitor='val_loss',
    #                     mode='min',
    #                     save_best_only=True,
    #                     save_weights_only=True,
    #                     verbose=0),
    #     EarlyStopping(monitor='accuracy', patience=1)
    # ]

    callbacks_list = [
        ModelCheckpoint("./results/keras_test.model",
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=0)
    ]

    print("processing fold: ", fold_ + 1)
    logging.info('mean_time={}'.format(str(fold_ + 1)))

    # history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=callbacks_list)

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
    model.load_weights('./results/keras_test.model')
    # serialize model to JSON
    model_json = model.to_json()
    with open("./results/segment_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./results/segment_model.h5")
    print("Saved model to disk")
    # Get predicted probabilities for each class
    y_pred_raw = model.predict(testX, batch_size=batch_size)

    y_pred_resh = y_pred_raw.reshape(y_pred_raw.shape[0], y_pred_raw.shape[2], -1)
    y_pred_resh_argmax = np.argmax(y_pred_resh, axis=2)
    y_pred = y_pred_resh_argmax.reshape(y_pred_resh_argmax.size)

    testy_resh = testy.reshape(testy.shape[0], testy.shape[2], -1)
    testy_resh_argmax = np.argmax(testy_resh, axis=2)
    labels_test_unary = testy_resh_argmax.reshape(testy_resh_argmax.size)

    oof_preds[test_, :] = y_pred_resh

    print('MULTI WEIGHTED LOG LOSS : %.5f '
          % common.multi_weighted_logloss(np.reshape(testy,
                                                     (testy.shape[0] * testy.shape[2], testy.shape[3])),
                                          np.reshape(y_pred_raw,
                                                     (y_pred_raw.shape[0] * y_pred_raw.shape[2], y_pred_raw.shape[3]))))

    clfs.append(model)
    label_index = list(range(1, act_classes + 1))

    accuracy, precision, recall, fscore, fw = common.checkAccuracy(labels_test_unary + 1,
                                                                   y_pred.reshape(y_pred.size) + 1,
                                                                   label_index)

    accuracy_folds.append(accuracy)
    precision_folds.append(precision)
    recall_folds.append(recall)
    fscore_folds.append(fscore)
    fw_folds.append(fw)


# selecting the best model
model_num = np.argmax(accuracy)
model = clfs[model_num]

# print model summary
print('model summary:')
print(str(model.summary()))

logging.info("model summary:")
model.summary(print_fn=logging.info)


print('__________________________________________________________________________________________________')
print('model performance per fold and showing results:')

print('avg (SD) accuracy: ' + str(mean(accuracy_folds)) + '(' + str(std(accuracy_folds)) + ')')
print('avg (SD) precision: ' + str(mean(precision_folds)) + '(' + str(std(precision_folds)) + ')')
print('avg (SD) recall: ' + str(mean(recall_folds)) + '(' + str(std(recall_folds)) + ')')
print('avg (SD) fscore: ' + str(mean(fscore_folds)) + '(' + str(std(fscore_folds)) + ')')
print('avg (SD) fw: ' + str(mean(fw_folds)) + '(' + str(std(fw_folds)) + ')')

logging.info("tmodel performance per fold and showing results:")
logging.info('avg (SD) accuracy:{}'.format(str(mean(accuracy_folds)) + ' (' + str(std(accuracy_folds)) + ')'))
logging.info('avg (SD) precision:{}'.format(str(mean(precision_folds)) + ' (' + str(std(precision_folds)) + ')'))
logging.info('avg (SD) recall:{}'.format(str(mean(recall_folds)) + ' (' + str(std(recall_folds)) + ')'))
logging.info('avg (SD) fscore:{}'.format(str(mean(fscore_folds)) + ' (' + str(std(fscore_folds)) + ')'))
logging.info('avg (SD) fw:{}'.format(str(mean(fw_folds)) + ' (' + str(std(fw_folds)) + ')'))

print('__________________________________________________________________________________________________')
print('testing the model and showing results on best model:')

start = time.perf_counter()
model.evaluate(X_test, y_test, batch_size=batch_size)
end = time.perf_counter()
sum_time += (end - start)
print('time passed for testing: ', str(end - start), 'sec')
logging.info('mean_time={}'.format(str(end - start)))

y_pred_prob = model.predict(X_test, batch_size=batch_size)

y_pred_resh = y_pred_prob.reshape(y_pred_prob.shape[0], y_pred_prob.shape[2], -1)
y_pred_resh_argmax = np.argmax(y_pred_resh, axis=2)
y_pred = y_pred_resh_argmax.reshape(y_pred_resh_argmax.size)

y_test_resh = y_test.reshape(y_test.shape[0], y_test.shape[2], -1)
y_test_resh_argmax = np.argmax(y_test_resh, axis=2)
labels_test_unary = y_test_resh_argmax.reshape(y_test_resh_argmax.size)

print('prediction results data shape: ', y_pred_prob.shape)
file_y_pred = './results/' + 'y_pred_' + args.dataset + '_' + \
              str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_y_pred, y_pred)
file_y_pred_prob = './results/' + 'y_pred_prob_' + args.dataset + '_' + \
                   str(subseq) + '_' + args.net + '_' + args.block + '.npy'
np.save(file_y_pred_prob, y_pred_prob)

label_index = list(range(1, act_classes + 1))
accuracy, precision, recall, fscore, fw = common.checkAccuracy(labels_test_unary + 1,
                                                               y_pred.reshape(y_pred.size) + 1,
                                                               label_index)
print("testing confusion matrix:")
print(common.createConfusionMatrix(labels_test_unary + 1,
                                   y_pred.reshape(y_pred.size) + 1,
                                   label_index))

logging.info("testing confusionmatrix:")
logging.info('testing confusionmatrix:{}'.format(common.createConfusionMatrix(labels_test_unary + 1,
                                                                              y_pred.reshape(y_pred.size) + 1,
                                                                              label_index)))

print('testing acc:{}'.format(accuracy))
logging.info('testing acc:{}'.format(accuracy))
print('testing fscore:{}'.format(fscore))
logging.info('testing fscore:{}'.format(fscore))
print('testing weighted fscore:{}'.format(fw))
logging.info('testing weighted fscore:{}'.format(fw))

# plot results
print("saving the confusion matrix graph")
common.plot_confusion_matrix(common.createConfusionMatrix(labels_test_unary + 1,
                                                          y_pred.reshape(y_pred.size) + 1,
                                                          label_index),
                             plot_classes=class_names,
                             normalize=True,
                             title='Confusion matrix')
print("saving the history result graphs")
common.plot_loss_acc(history)

print("results saved in _results_ folder")

end_total = time.perf_counter()
print('time passed for the whole process: ', str(end_total - start_total), 'sec')
logging.info('mean_time_total={}'.format(str(end_total - start_total)))


