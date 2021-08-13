# This file contains some data handling routines ##
# Authors: Hariharan Seshadri, Keerthi Nagaraj, Nitin Jain ##

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from collections import defaultdict
import scipy
import csv
import itertools
from itertools import cycle

###################################################################################
# Given a file name, returns the parsed file in the form of an array ##

def parseFile(file_name):
    f = open(file_name)
    featureArray = []
    lines = f.readlines()
    for line in lines:
        feature_length = len(line.split(" "))

        raw_feature = line.split(" ")

        feature = []

        for index in range(feature_length):
            try:
                feature.append(float(raw_feature[index]))
            except:
                continue

        featureArray.append(feature)

    return np.asarray(featureArray)


def parseCSVFile(file_name):
    f = open(file_name)
    featureArray = []
    reader = csv.reader(f)
    for row in reader:
        feature_length = len(row)

        # raw_feature = line.split(" ")

        feature = []

        for index in range(feature_length):
            try:
                feature.append(float(row[index]))
            except:
                continue

        featureArray.append(feature)

    return np.asarray(featureArray)


# reader = csv.reader(csvfile)
# rows = [row for row in reader]	

###################################################################################
# Given two LISTS- original and predicted , returns the precision, accuracy, fscore

def checkAccuracy(original, predicted, labels):
    TP = defaultdict(list)
    TN = defaultdict(list)
    FP = defaultdict(list)
    FN = defaultdict(list)
    cnt = defaultdict(list)
    ctrue = 0
    cfalse = 0
    w = []
    num = 0
    accuracy = []
    precision = []
    recall = []
    f_score = []
    fw = 0.0
    for i in range(len(original)):
        cnt[str(int(original[i]))].append(1)
        if original[i] == predicted[i]:
            TP[str(int(original[i]))].append(1)
            TN[str(int(predicted[i]))].append(1)
            ctrue = ctrue + 1

        elif original[i] != predicted[i]:
            FP[str(int(predicted[i]))].append(1)
            FN[str(int(original[i]))].append(1)
            cfalse = cfalse + 1

    for label in labels:
        if (len(TP[str(label)]) + len(FP[str(label)])) != 0:
            p = float(len(TP[str(label)])) / (len(TP[str(label)]) + len(FP[str(label)]))
        else:
            p = 0.0
        precision.append(p)
        if (len(TP[str(label)]) + len(FN[str(label)])) != 0:
            r = float(len(TP[str(label)])) / (len(TP[str(label)]) + len(FN[str(label)]))
        else:
            r = 0.0
        recall.append(r)
        if (p + r) != 0.0:
            fs = float(2 * p * r) / (p + r)
        else:
            fs = 0.0
        f_score.append(fs)

        num = num + len(cnt[str(label)])
    for label in labels:
        ww = float(len(cnt[str(label)])) / (num)
        # fw=fw+float(f[str(label)])*(ww)/(num)
        w.append(ww)
    for i in range(len(w)):
        fw = fw + w[i] * f_score[i]
    # a = float( len(TP)+len(TN) )/ ( len(TP) + len(FP)+len(TN)+len(FN))
    accuracy = float(ctrue) / (ctrue + cfalse)
    return accuracy, precision, recall, f_score, fw


###################################################################################

# Distinguishes labels as Dynamic[1]/Non-Dynamic[0] ##

def convertLabel(labels, posLabels, Neglabels):
    dynamic = []

    for label in labels:

        if label in posLabels:
            dynamic.append(1)

        elif label in Neglabels:
            dynamic.append(0)
        else:
            print("Unknown Label: Good Gawd :)")
    return np.asarray(dynamic)


###################################################################################
# This function takes in input 2D array of inputData(X) and 1D array of inputLabels(Y) and returns
# subset of those data which belong to requiredLabels(Ex: [1,4,5]). Required Labels is a 1D list.
# Returns 2D array of subData (X'), 1D array of subLabels(Y')
def getDataSubset(inputData, inputLabels, RequiredLabels):
    subData = []
    subLabels = []
    for loopVar in range(len(inputLabels)):
        if inputLabels[loopVar] in RequiredLabels:
            subData.append(inputData[loopVar])
            subLabels.append(inputLabels[loopVar])
    return np.asarray(subData), np.asarray(subLabels)


###################################################################################

# This function creates the Confusion Matrix for the predicted model
def createConfusionMatrix(predictedYLabels, originalYLabels, labelList):
    confusionMatrix = np.zeros((len(labelList), len(labelList)))
    # print len(predictedYLabels)

    if len(originalYLabels) != len(predictedYLabels):
        print('Error')
        return

    for i in range(len(originalYLabels)):
        if (predictedYLabels[i] not in labelList) or (originalYLabels[i] not in labelList):
            print('Error')
            return
        else:
            confusionMatrix[labelList.index(originalYLabels[i]), labelList.index(predictedYLabels[i])] = \
                confusionMatrix[labelList.index(originalYLabels[i]), labelList.index(predictedYLabels[i])] + 1
    return confusionMatrix


#############################################################################

# This function returns the Mahalanobis distance between two given class 1 TO class 2 with
# respect to class 1's variance (i.e.) Mahalanobis Distance). NOTE: labels is a LIST containing only TWO labels.

def getMahalanobisDistance(X_train, Y_train, labels):
    labelA = labels[0]
    labelB = labels[1]

    X_A, Y_A = getDataSubset(X_train, Y_train, [labelA])
    mean_A = np.mean(X_A, axis=0)
    cov_A = np.cov(X_A, rowvar=0)

    X_B, Y_B = getDataSubset(X_train, Y_train, [labelB])
    mean_B = np.mean(X_B, axis=0)
    cov_B = np.cov(X_B, rowvar=0)

    return scipy.spatial.distance.mahalanobis(mean_A, mean_B, cov_A)


#############################################################################

# This function returns the Mean and Covariance of tha data of a particular label. 'label' requires a single number

def getDistribution(X_train, Y_train, label):
    X_A, Y_A = getDataSubset(X_train, Y_train, [label])
    mean_A = np.mean(X_A, axis=0)
    cov_A = np.cov(X_A, rowvar=0)

    return mean_A, cov_A


#############################################################################

# This function returns the sample weights based on HOW CLOSE THE SAMPLE IS TO THE MEAN
# OF IT'S CLASS , "labels" is a LIST that specifies the labels in Y_train

def getSampleWeights(X_train, Y_train, labels):
    sample_weights = []
    mean = []
    cov = []

    for i in labels:
        mean_A, cov_A = getDistribution(X_train, Y_train, i)

        mean.append(mean_A)

        cov.append(cov_A)

    for i in range(len(X_train)):
        index = labels.index(int(Y_train[i]))
        this_mean = mean[index]
        this_cov = cov[index]

        weight = scipy.spatial.distance.mahalanobis(X_train[i], this_mean, this_cov)

        weight = float(1) / weight

        sample_weights.append(weight)

    return np.asarray(sample_weights)


#############################################################################

# This function returns the the input array whose entries are raised by power list 'k'.

def getPowerK(X_features, k):
    X_features_new = []

    for x_feature in X_features:
        x_feature_new = []

        for power in k:
            for x_feature_dimension in x_feature:
                x_feature_new.append(np.power(x_feature_dimension, power))

        X_features_new.append(np.asarray(x_feature_new))

    return np.asarray(X_features_new)


#############################################################################
# returns a validation and training dataset from the full datatset sent as input parameters

def getValidationDataset(X_full, Y_full, labels=[1, 2, 3, 4, 5, 6], splitRatio=3):
    fullDatasetSize = 7352
    if (len(X_full) != len(Y_full)) and (len(Y_full) != fullDatasetSize):
        print("Error: Not the full dataset or X and Y are unequal")
        return
    else:
        indexLists = dict()
        for i in labels:
            indexLists[i] = []
        for j in range(fullDatasetSize):
            if Y_full[j] in labels:
                indexLists[Y_full[j]].append(j)
        for i in labels: print(len(indexLists[i]))

        X_d = []
        Y_d = []
        X_v = []
        Y_v = []
        for j in labels:
            datasetSizeforLabel = len(indexLists[j])
            ValidationDatasetSizeforLabel = datasetSizeforLabel / (splitRatio + 1)
            print(datasetSizeforLabel, ValidationDatasetSizeforLabel)
            taken = []
            for i in range(ValidationDatasetSizeforLabel):
                from random import randint
                while True:
                    rand = randint(0, datasetSizeforLabel)
                    if rand not in taken:
                        taken.append(rand)
                        break
            print(taken)
            print(indexLists[j])
            cnt = 0
            for i in range(datasetSizeforLabel):
                if i in taken:
                    cnt = cnt + 1
                    X_v.append(X_full[indexLists[j][i], :])
                    Y_v.append(Y_full[indexLists[j][i]])
                else:
                    X_d.append(X_full[indexLists[j][i], :])
                    Y_d.append(Y_full[indexLists[j][i]])
            print(cnt)

    # return np.asarray(X_v),np.asarray(Y_v),np.asarray(X_d),np.asarray(Y_d)
    f = open('X_Validation.txt', 'w+')
    # f.write(X_v)
    np.savetxt(f, X_v)
    # np.save(f,np.asarray(X_v))
    f.close()
    f = open('Y_Validation.txt', 'w+')
    np.savetxt(f, Y_v)
    # np.save(f,np.asarray(Y_v))
    f.close()
    f = open('X_training.txt', 'w+')
    # np.save(f,np.asarray(X_d))
    # f.write(X_d)
    np.savetxt(f, X_d)
    f.close()
    f = open('Y_training.txt', 'w+')
    # np.save(f,np.asarray(Y_d))
    # f.write(Y_d)
    np.savetxt(f, Y_d)
    f.close()


###################################################################################
# Returns the parsed file in the form of an array containing only Accelero features##

def getAccFeatures(X_train, features_file='../UCI HAR Dataset/features.txt'):
    f = open(features_file)
    lines = f.readlines()
    AccFeaturesList = []
    i = 0
    for line in lines:
        if not 'Gyro' in line: AccFeaturesList.append(i)
        i = i + 1
    f.close()

    features = []
    for index in AccFeaturesList:
        features.append(X_train[:, index])

    return np.transpose(np.asarray(features))


###################################################################################
## Returns the parsed file in the form of an array containing only Gyro features##

def getGyroFeatures(X_train, feature_file='../UCI HAR Dataset/features.txt'):
    f = open(feature_file)
    lines = f.readlines()
    GyroFeaturesList = []
    i = 0
    for line in lines:
        if not 'Acc' in line: GyroFeaturesList.append(i)
        i = i + 1
    f.close()

    features = []
    for index in GyroFeaturesList:
        features.append(X_train[:, index])
    return np.transpose(np.asarray(features))


##################################################################################
# returns data for requested subjects
def getSubjectData(inputXData, inputYData, requiredSubjects, subjectData=None):
    requiredSubjectDataIndexList = []
    if subjectData is None:
        subjectData = parseFile('../UCI HAR Dataset/train/subject_train.txt')

    for i in range(len(subjectData)):
        if int(subjectData[i]) in requiredSubjects:
            requiredSubjectDataIndexList.append(i);
    return inputXData[requiredSubjectDataIndexList, :], inputYData[requiredSubjectDataIndexList], subjectData[
        requiredSubjectDataIndexList]


##################################################################################
# plot results of loss and accuracy over epochs

def plot_loss_acc(history_results):
    fig = plt.figure()
    plt.plot(history_results.history['loss'])
    plt.plot(history_results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.ioff()
    plt.savefig('./results/graphs/loss_history.png')
    # plt.show()

    fig = plt.figure()
    plt.plot(history_results.history['accuracy'])
    plt.plot(history_results.history['val_accuracy'])
    plt.title('model Accuracy')
    plt.ylabel('val_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.ioff()
    plt.savefig('./results/graphs/accuracy_history.png')
    # plt.show()


##################################################################################
# plot the confusion matrix
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, plot_classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(5, 5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(plot_classes))
    plt.xticks(tick_marks, plot_classes, rotation=45)
    plt.yticks(tick_marks, plot_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(jj, ii, format(cm[ii, jj], fmt),
                 horizontalalignment="center",
                 color="white" if cm[ii, jj] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.ioff()
    plt.savefig('./results/graphs/confusion_matrix.png')


##################################################################################
# plot the confusion matrix
def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    data_classes = list(range(y_p.shape[1]))
    data_class_weight = {b: a + 1 for a, b in enumerate(set(data_classes) - {'<'})}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([data_class_weight[k] for k in sorted(data_class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


##################################################################################
# plot ROC curve

# def plot_roc(fpr_list, tpr_list, auc_list):
#     plt.figure(figsize=(12, 7))
#     for i in range(0, len(fpr_list)):
#         plt.plot(fpr_list[i], tpr_list[i], label=f'AUC (Trial) = {auc_list[i]:.2f}')
#     plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
#     plt.title('ROC Curve', size=20)
#     plt.xlabel('False Positive Rate', size=14)
#     plt.ylabel('True Positive Rate', size=14)
#     plt.legend()
#     plt.ioff()
#     plt.savefig('./results/graphs/auc_history.png')

def plot_roc(fpr, tpr, roc_auc, n_classes):
    lw = 2
    plt.figure(figsize=(12, 7))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.title('Receiver operating characteristic curves', size=20)
    plt.legend(loc="lower right")
    plt.ioff()
    plt.savefig('./results/graphs/auc_history.png')


def plot_roc_one_class(fpr, tpr, roc_auc, i):
    plt.figure(figsize=(12, 7))
    lw = 2
    plt.plot(fpr[i], tpr[i], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.title('Receiver operating characteristic curve', size=20)
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc="lower right")
    plt.ioff()
    plt.savefig('./results/graphs/auc_history_one_class.png')


def plot_precision_recall_all_classes(recall, precision, average_precision):
    plt.figure(figsize=(12, 7))
    plt.step(recall["micro"], precision["micro"], color='darkorange', where='post')
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
              .format(average_precision["micro"]), size=20)
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.ioff()
    plt.savefig('./results/graphs/precision_recall_all_classes.png')


def plot_precision_recall(recall, precision, average_precision, n_classes):
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(12, 7))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.title('Extension of Precision-Recall curve to multi-class', size=20)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.ioff()
    plt.savefig('./results/graphs/precision_recall.png')


