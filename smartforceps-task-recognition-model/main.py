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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras import backend as K
import pickle
# import unet_224_model
import model_lstm
import model_ineption
import data_load_task
import model_info
import common
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
file_log = './results/' + args.net + '_' + args.dataset + '_' + str(subseq) + '.log'

logging.basicConfig(filename=file_log, level=logging.DEBUG)
model_info.begin()

start_total = time.perf_counter()

read_processed_data = data_load_task.load_data(args.dataset, subseq=subseq)

X = read_processed_data[0]
y = read_processed_data[1]
X_train = read_processed_data[2]
X_val = read_processed_data[3]
X_test = read_processed_data[4]
y_train = read_processed_data[5]
y_val = read_processed_data[6]
y_test = read_processed_data[7]
y_map_train = read_processed_data[8]
wtable_train = read_processed_data[9]
y_map_test = read_processed_data[10]
N_FEATURES = read_processed_data[11]
y_train_raw = read_processed_data[12]
y_val_raw = read_processed_data[13]
act_classes = read_processed_data[14]
class_names = read_processed_data[15]


# user defined loss function
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true, y_pred):
    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.math.log(yc), axis=0) / wtable_train))
    return loss


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

clfs = []
oof_preds = np.zeros((y_train.shape[0], y_train.shape[1]))
epochs = 150
batch_size = 128
optim_type = 'adam'
sum_time = 0
hyper_params_list_len = 4
learning_rate_list = [0.001, 0.01, 0.1]
units_size_list = [100, 200, 400, 600]
depth_list = [6, 8, 10, 12]

acc_hist = []
val_acc_hist = []
loss_hist = []
val_loss_hist = []
model_hist = []
model_var_hist = []
for i in range(len(learning_rate_list)):
    acc_hist.append([])
    val_acc_hist.append([])
    loss_hist.append([])
    val_loss_hist.append([])
    model_hist.append([])
    model_var_hist.append([])
    for j in range(hyper_params_list_len):

        learning_rate = learning_rate_list[i]
        print('learning rate = ', learning_rate)

        trainX, trainy = X_train, y_train
        validX, validy = X_val, y_val

        if args.net == 'lstm_model':
            units_size_val = units_size_list[j]
            print('network units = ', units_size_val)
            sub_model = model_lstm.lstm_model(units_size=units_size_val,
                                              timesteps_count=subseq,
                                              feature_count=N_FEATURES,
                                              activation='relu',
                                              n_output=act_classes)
        elif args.net == 'inception_time':
            depth_val = depth_list[j]
            print('network depth = ', depth_val)
            sub_model = model_ineption.build_inception_model(
                input_shape=trainX.shape[1:],
                nb_classes=act_classes,
                depth=depth_val)

        if optim_type == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optim = Adam(lr=learning_rate)

        sub_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

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
            ModelCheckpoint("./results/keras_task_" + str(i) + str(j) + ".model",
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=0)
        ]

        history = sub_model.fit(trainX, trainy,
                                validation_data=(validX, validy),
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=True,
                                # verbose=0,
                                callbacks=callbacks_list)

        sub_acc = np.array(history.history['accuracy'])
        sub_loss = np.array(history.history['loss'])
        sub_val_acc = np.array(history.history['val_accuracy'])
        sub_val_loss = np.array(history.history['val_loss'])

        acc_hist[i].append(sub_acc)
        loss_hist[i].append(sub_loss)
        val_acc_hist[i].append(sub_val_acc)
        val_loss_hist[i].append(sub_val_loss)
        model_hist[i].append(sub_model)
        model_var_hist[i].append(history)

# save history of validation losses
np.save('./results/val_loss_history.npy', val_loss_hist, allow_pickle=True)

# save history of validation acc
np.save('./results/val_acc_history.npy', val_acc_hist, allow_pickle=True)

# find minimum validation loss across iterations
min_val_loss_idx = np.where(val_loss_hist == np.amin(val_loss_hist))
print('min validation loss idx: ', min_val_loss_idx)
logging.info('min validation loss idx: {}'.format(", ".join(str(x) for x in min_val_loss_idx)))

# select min loss across iterations
acc = acc_hist[min_val_loss_idx[0].tolist()[0]][min_val_loss_idx[1].tolist()[0]]
loss = loss_hist[min_val_loss_idx[0].tolist()[0]][min_val_loss_idx[1].tolist()[0]]
val_acc = val_acc_hist[min_val_loss_idx[0].tolist()[0]][min_val_loss_idx[1].tolist()[0]]
val_loss = val_loss_hist[min_val_loss_idx[0].tolist()[0]][min_val_loss_idx[1].tolist()[0]]
model = model_hist[min_val_loss_idx[0].tolist()[0]][min_val_loss_idx[1].tolist()[0]]
history = model_var_hist[min_val_loss_idx[0].tolist()[0]][min_val_loss_idx[1].tolist()[0]]

for i in range(acc.shape[0]):
    logging.info(
        'Epoch: {} loss: {:.4f} accuracy{:.4f} val_loss: {:.4f} val_accuracy{:.4f}%\n'.format(i + 1, loss[i],
                                                                                              acc[i],
                                                                                              val_loss[i],
                                                                                              val_acc[i]))

print('Loading Best Model')
model.load_weights('./results/keras_task_' + str(min_val_loss_idx[0].tolist()[0]) +
                   str(min_val_loss_idx[1].tolist()[0]) + '.model')
# serialize model to JSON
model_json = model.to_json()
with open("./results/task_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./results/task_model.h5")
print("Saved model to disk")
# get predicted probabilities for each class
valid_pred = model.predict(validX, batch_size=batch_size)
oof_preds = valid_pred
print('MULTI WEIGHTED LOG LOSS : %.5f '
      % common.multi_weighted_logloss(validy, valid_pred))

clfs.append(model)

# print model summary
print('model summary:')
print(str(model.summary()))
logging.info("model summary:")
model.summary(print_fn=logging.info)

# take the class with the highest probability from the train predictions
classification_report_var = classification_report(pd.DataFrame(y_val_raw),
                                                  np.argmax(oof_preds, axis=1))
print('classification report:\n', classification_report_var)
logging.info('classification report:{}\n'.format(classification_report_var))

print('MULTI WEIGHTED LOG LOSS : %.5f '
      % common.multi_weighted_logloss(y_val, oof_preds))

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

# ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(act_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ROC curve and ROC area for all classes
# aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(act_classes)]))
# interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(act_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# average it and compute AUC
mean_tpr /= act_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# multiclass area under ROC
macro_roc_auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class="ovo", average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
logging.info("One-vs-One ROC AUC scores:")
logging.info('{:.6f} (macro)'.format(macro_roc_auc_ovr))
logging.info('{:.6f} (weighted by prevalence)'.format(weighted_roc_auc_ovr))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
logging.info("One-vs-Rest ROC AUC scores:")
logging.info('{:.6f} (macro)'.format(macro_roc_auc_ovr))
logging.info('{:.6f} (weighted by prevalence)'.format(weighted_roc_auc_ovr))

# Precision-Recall curve for each class and average precision score
precision_dict = dict()
recall_dict = dict()
average_precision = dict()
for i in range(act_classes):
    precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test[:, i],
                                                                  y_pred_prob[:, i])
    average_precision[i] = average_precision_score(y_test[:, i],
                                                   y_pred_prob[:, i])

# a "micro-average": quantifying score on all classes jointly
precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                          y_pred_prob.ravel())
average_precision["micro"] = average_precision_score(y_test,
                                                     y_pred_prob,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))
logging.info("Average precision score:")
logging.info('{:.6f} (micro-averaged over all classes)'.format(average_precision["micro"]))

# Create confusion matrix
label_index = list(range(1, act_classes + 1))
accuracy, precision, recall, fscore, fw = common.checkAccuracy(labels_test_unary + 1, y_pred + 1, label_index)
print("testing confusion matrix:")
print(common.createConfusionMatrix(labels_test_unary + 1, y_pred + 1, label_index))

logging.info("testing confusionmatrix:")
logging.info('{}'.format(common.createConfusionMatrix(labels_test_unary + 1,
                                                      y_pred + 1,
                                                      label_index)))

print('testing acc:{}'.format(accuracy))
logging.info('testing acc:{}'.format(accuracy))
print('testing fscore:{}'.format(fscore))
logging.info('testing fscore:{}'.format(fscore))
print('testing weighted fscore:{}'.format(fw))
logging.info('testing weighted fscore:{}'.format(fw))

# Plot results
print("saving the confusion matrix graph")
common.plot_confusion_matrix(common.createConfusionMatrix(labels_test_unary + 1, y_pred + 1, label_index),
                             plot_classes=class_names,
                             normalize=True,
                             title='Confusion matrix')
print("saving the history result graphs")
common.plot_loss_acc(history)

print("saving the auc result graphs")
common.plot_roc(fpr, tpr, roc_auc, act_classes)
roc_dicts = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'act_classes': act_classes}
with open('./results/' + 'roc_dicts.pkl', 'wb') as f:
    pickle.dump(roc_dicts, f, pickle.HIGHEST_PROTOCOL)

print("saving the auc result graphs for class 0 (coagulation)")
common.plot_roc_one_class(fpr, tpr, roc_auc, 0)

print("saving the precision recall graph for all classes")
common.plot_precision_recall_all_classes(recall_dict, precision_dict, average_precision)

print("saving the precision recall graph for each class")
common.plot_precision_recall(recall_dict, precision_dict, average_precision, act_classes)

print("results saved in _results_ folder")

end_total = time.perf_counter()
print('time passed for the whole process: ', str(end_total - start_total), 'sec')
logging.info('mean_time_total={}'.format(str(end_total - start_total)))
