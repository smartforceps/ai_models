import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from collections import defaultdict
import scipy
import csv
import itertools
from itertools import cycle


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

##################################################################################

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


##################################################################################
# plot results of loss and accuracy over epochs


def plot_loss_acc(history_results):
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    # colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    # colors = [plt.cm.tab20b(i / float(5)) for i in range(5)]

    fig = plt.figure()
    plt.plot(history_results.history['loss'], color=colors[0])
    plt.plot(history_results.history['val_loss'], color=colors[1])
    plt.title('Model Loss')
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ioff()
    plt.savefig('./results/graphs/loss_history.png')
    # plt.show()

    fig = plt.figure()
    plt.plot(history_results.history['accuracy'], color=colors[0])
    plt.plot(history_results.history['val_accuracy'], color=colors[1])
    plt.title('Model Accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.ioff()
    plt.savefig('./results/graphs/accuracy_history.png')
    # plt.show()


##################################################################################
# plot the confusion matrix
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, plot_classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="BuGn"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(5, 5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap, alpha=0.5)
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

def plot_roc(fpr, tpr, roc_auc, n_classes):
    lw = 2
    plt.figure(figsize=(12, 7))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color="orange", linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='g', linestyle=':', linewidth=2)

    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    # colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    # colors = [plt.cm.tab20b(i / float(5)) for i in range(5)]
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
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
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    # colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    # colors = [plt.cm.tab20b(i / float(5)) for i in range(5)]
    plt.figure(figsize=(12, 7))
    lw = 2
    # plt.plot(fpr[i], tpr[i], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot(fpr[i], tpr[i], color=colors[0], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.title('Receiver operating characteristic curve', size=20)
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.ioff()
    plt.savefig('./results/graphs/auc_history_one_class.png')


def plot_precision_recall_all_classes(recall, precision, average_precision):
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    # colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    # colors = [plt.cm.tab20b(i / float(5)) for i in range(5)]
    plt.figure(figsize=(12, 7))
    # plt.step(recall["micro"], precision["micro"], color='darkorange', where='post')
    plt.step(recall["micro"], precision["micro"], color=colors[0], where='post')
    plt.title('Average precision score, micro-averaged over all classes: Average Precision={0:0.2f}'
              .format(average_precision["micro"]), size=20)
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.ioff()
    plt.savefig('./results/graphs/precision_recall_all_classes.png')


def plot_precision_recall(recall, precision, average_precision, n_classes):
    # setup plot details
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    # colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    # colors = [plt.cm.tab20b(i / float(5)) for i in range(5)]
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

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
    plt.title('Extension of Precision-Recall curve to multi-class', size=24)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.ioff()
    plt.savefig('./results/graphs/precision_recall.png')