# encoding=utf8

"""
    model for smartforceps task data classification

"""

import os
import numpy as np
import pandas as pd
from numpy import stack
from tensorflow.keras.utils import to_categorical
import h5py
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from sklearn.model_selection import cross_validate

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

import xgboost as xgb
from xgboost import XGBClassifier

from statistics import mean
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import skew, kurtosis, normaltest

from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects as ro
from skimage.transform import resize

RANDOM_SEED = 42

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100


def bw_filter(signal_in):
    fs = 20  # Sampling frequency
    fc = 1  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(4, w, 'low')
    output = signal.filtfilt(b, a, signal_in)
    return output


def feature_normalization(X):
    X = X.astype(float)
    scaler = StandardScaler().fit(X)
    data = scaler.transform(X)
    return data


def df_resample(df1, num=1):
    df2 = pd.DataFrame()
    for key, value in df1.iteritems():
        temp = value.to_numpy() / value.abs().max()  # normalize
        resampled = resize(temp, (num, 1), mode='edge') * value.abs().max()  # de-normalize
        df2[key] = resampled.flatten().round(2)
    return df2


def py_tsfeatures(df):
    df_result = pd.DataFrame(data={'data': [0]})

    # df_result['Duration.Force'] = round((max(df['MillisecondsSinceRecord']) -
    #                                      min(df['MillisecondsSinceRecord'])) / 1000, 4)
    #
    # df_result['Mean.Force'] = round(mean([df['LeftCalibratedForceValue'].mean(),
    #                                       df['RightCalibratedForceValue'].mean()]), 4)
    #
    # df_result['Max.Force'] = round(max(df['LeftCalibratedForceValue'].max(),
    #                                    df['RightCalibratedForceValue'].max()), 4)
    #
    # df_result['Min.Force'] = round(min(df['LeftCalibratedForceValue'].min(),
    #                                    df['RightCalibratedForceValue'].min()), 4)
    #
    # df_result['Range.Force'] = round(df['LeftCalibratedForceValue'].max() -
    #                                  df['RightCalibratedForceValue'].min(), 4)
    #
    df_result['Median.Force'] = round(mean([df['LeftCalibratedForceValue'].median(),
                                            df['RightCalibratedForceValue'].median()]), 4)
    #
    # df_result['SD.Force'] = round(max(df['LeftCalibratedForceValue'].std(),
    #                                   df['RightCalibratedForceValue'].std()), 4)
    #
    # df_result['Coef.Variance'] = round(max(df[['LeftCalibratedForceValue']].apply(cv)[0],
    #                                        df[['RightCalibratedForceValue']].apply(cv)[0]), 4)
    #
    # df_result['Skewness'] = round(max([abs(skew(df['LeftCalibratedForceValue'])),
    #                                    abs(skew(df['RightCalibratedForceValue']))]), 4)
    #
    # df_result['Kurtosis'] = round(max([abs(kurtosis(df['LeftCalibratedForceValue'])),
    #                                    abs(kurtosis(df['RightCalibratedForceValue']))]), 4)
    #
    # df_result['Normtest'] = round(min(normaltest(df['LeftCalibratedForceValue']).pvalue,
    #                                   normaltest(df['RightCalibratedForceValue']).pvalue), 4)
    #
    # peaks_left, properties_left = find_peaks(bw_filter(df['LeftCalibratedForceValue'].to_numpy()),
    #                                          distance=5,
    #                                          prominence=0.5,
    #                                          wlen=10)
    # peaks_right, properties_right = find_peaks(bw_filter(df['RightCalibratedForceValue'].to_numpy()),
    #                                            distance=5,
    #                                            prominence=0.5,
    #                                            wlen=10)
    #
    # try:
    #     df_result['Peaks.Count'] = len(set(peaks_left.tolist() + peaks_right.tolist()))
    #
    #     df_result['Max.Peak.Value'] = max(max((df['LeftCalibratedForceValue'].iloc[peaks_right] -
    #                                            df['LeftCalibratedForceValue'].mean()).to_list()),
    #                                       max((df['RightCalibratedForceValue'].iloc[peaks_right] -
    #                                            df['RightCalibratedForceValue'].mean()).to_list()))
    # except:
    #     df_result['Peaks.Count'] = 0
    #     df_result['Max.Peak.Value'] = np.nan
    #
    # df_result['Frequency'] = (df_result[['Peaks.Count']].to_numpy()[0][0] + 1) / \
    #                          df_result[['Duration.Force']].to_numpy()[0][0]
    #
    # df_result['Period.Length'] = df_result[['Duration.Force']].to_numpy()[0][0] / (
    #         df_result[['Peaks.Count']].to_numpy()[0][0] + 1)
    #
    ts_features_left = TsFeatures().transform(TimeSeriesData(df[['MillisecondsSinceRecord',
                                                                 'LeftCalibratedForceValue']].rename(
        columns={'MillisecondsSinceRecord': 'time'})))

    ts_features_right = TsFeatures().transform(TimeSeriesData(df[['MillisecondsSinceRecord',
                                                                  'RightCalibratedForceValue']].rename(
        columns={'MillisecondsSinceRecord': 'time'})))
    #
    # df_result['Diff.Force.SD'] = round(max(ts_features_left['std1st_der'],
    #                                        ts_features_right['std1st_der']), 4)
    #
    # df_result['Flat.Spots'] = round(min(ts_features_left['flat_spots'],
    #                                     ts_features_right['flat_spots']), 4)
    #
    df_result['Trend.Strength'] = round(mean([ts_features_left['trend_strength'],
                                              ts_features_right['trend_strength']]), 4)
    #
    # df_result['Linearity'] = round(max([abs(ts_features_left['linearity']),
    #                                     abs(ts_features_right['linearity'])]), 4)
    #
    # df_result['Stability'] = round(mean([ts_features_left['stability'],
    #                                      ts_features_right['stability']]), 4)
    #
    # df_result['Lumpiness'] = round(mean([ts_features_left['lumpiness'],
    #                                      ts_features_right['lumpiness']]), 4)
    #
    df_result['Crossing.Points'] = round(mean([ts_features_left['crossing_points'],
                                               ts_features_right['crossing_points']]), 4)
    #
    df_result['Entropy'] = round(max(ts_features_left['entropy'],
                                     ts_features_right['entropy']), 4)
    #
    # df_result['Heterogeneity'] = round(max(ts_features_left['heterogeneity'],
    #                                        ts_features_right['heterogeneity']), 4)
    #
    # df_result['Spikiness'] = round(max(ts_features_left['spikiness'],
    #                                    ts_features_right['spikiness']), 4)
    #
    df_result['First.Min.Autocorr'] = round(min(ts_features_left['firstmin_ac'],
                                                ts_features_right['firstmin_ac']), 4)
    #
    # df_result['First.Zero.Autocorr'] = round(min(ts_features_left['firstzero_ac'],
    #                                              ts_features_right['firstzero_ac']), 4)
    #
    # df_result['Autocorr.Function.1'] = round(min(ts_features_left['y_acf1'],
    #                                              ts_features_right['y_acf1']), 4)
    #
    # df_result['Autocorr.Function.5'] = round(min(ts_features_left['y_acf5'],
    #                                              ts_features_right['y_acf5']), 4)
    #
    df_result = df_result.drop(columns=['data'])

    return df_result


def prepdare_model_data(df, window_size, label_column):
    df.replace({label_column: {'Coagulation': 0,
                               'Other': 1,
                               'Pulling': 1,
                               'Manipulation': 2,
                               'Dissecting': 3,
                               'Retracting': 4,
                               'Novice': 0,
                               'Expert': 1
                               }}, inplace=True)

    df_segment = list()
    df_labels = list()
    for seg in range(int(df.shape[0] / window_size)):
        try:
            df_seg = df.iloc[(window_size * seg):(window_size * (seg + 1))][['LeftCalibratedForceValue',
                                                                             'RightCalibratedForceValue']]

            # add hand crafted features
            df_seg['MillisecondsSinceRecord'] = np.linspace(0, 50 * (window_size - 1), window_size)
            pd_df_results = py_tsfeatures(df_seg).replace(np.nan, 0)
            # df_seg = df_seg.drop(['MillisecondsSinceRecord'], axis=1)
            #
            # df_seg['RresampledFeatures'] = feature_normalization(df_resample(pd_df_results.T,
            #                                                                  window_size).to_numpy())

            # add hand crafted features

            df_segment.append(pd_df_results.values.tolist()[0])
            df_labels.append(int(df.iloc[(window_size * seg): (window_size * (seg + 1))][[label_column]].median()[0]))
        except:
            print('features not extractable')

    X, y = np.array(df_segment), np.array(df_labels)

    return X, y, pd_df_results.columns.tolist()


# classification_task = 'Skill'
classification_task = 'Task'

"""
Task Features for XGBoost:

Entropy	
Median Force	
Trend Strength	
First Min Autocorrelation	
Crossing Points

Skill Features for XGBoost:

Diff Force SD
Min Force
Crossing Points
Normtest

"""

if classification_task == 'Task':
    label_column = 'TaskCategory'

    # using balanced data
    os.chdir('..')
    df = pd.read_csv(os.getcwd() + '/data/df_force_data_with_label_balanced.csv', index_col=0, low_memory=False)
    os.chdir('data-analysis')

    label_original = ['Coagulation', 'Pulling', 'Manipulation', 'Dissecting', 'Retracting']
    label = ['Coagulation', 'Other']

    df.replace({'TaskCategory': {label_original[0]: label[0],
                                 label_original[1]: label[1],
                                 label_original[2]: label[1],
                                 label_original[3]: label[1],
                                 label_original[4]: label[1]}}, inplace=True)

elif classification_task == 'Skill':
    label_column = 'SkillClass'

    os.chdir('..')
    df = pd.read_csv(os.getcwd() + '/data/df_force_data_with_label.csv',
                     index_col=0, low_memory=False).iloc[:, [0, 1, 2, 6, 9, 11]]
    os.chdir('data-analysis')

    df = df.dropna()

window_size = 200
X, y, features_list = prepdare_model_data(df, window_size, label_column)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""

Permutation feature importance with knn for classification
"""

# define dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model_KNN = KNeighborsClassifier()
# fit the model
model_KNN.fit(x_train, y_train)
# perform permutation importance
results = permutation_importance(model_KNN, x_train, y_train, scoring='accuracy')
# get importance
importance_KNN = results.importances_mean
# summarize feature importance
for i, v in enumerate(importance_KNN):
    print('Feature: %0d, Score: %.5f' % (i, v))

df_importance_scores = pd.DataFrame({'Feature': features_list, 'Importance Score': importance_KNN})
df_importance_scores.sort_values(by=['Importance Score'], ascending=False, inplace=True)
df_importance_scores.to_csv('./smartforceps_' + classification_task.lower() + '_model/results/feature_based_model/' + classification_task +
                            '/KNeighborsClassifier_feature_importance.csv')

# plot feature importance
fig = plt.figure(figsize=(24, 10), dpi=320)
spacing = 0.2
fig.subplots_adjust(bottom=spacing)
plt.rcParams.update({'font.size': 11})  # must set in top
grid = plt.GridSpec(4, 4, hspace=0.8, wspace=0.2)
ax = sns.barplot(df_importance_scores['Feature'].to_list(),
                 df_importance_scores['Importance Score'].to_list())
ax.tick_params(axis='x', rotation=45)
ax.axes.set_title('Permutation Importance Comparison of Hand Crafted Features for ' + classification_task +
                  ' Classification', fontsize=22)
ax.set_xlabel('Feature', fontsize=18)
ax.set_ylabel('Importance Score', fontsize=18)
plt.savefig('./smartforceps_' +
            classification_task.lower() + '_model/'
            'results/feature_based_model/' + classification_task +
            '/KNeighborsClassifier_feature_importance.png', dpi=400)
plt.show()

"""

XGBoost Model for Time series Classification
"""

# Resolve overfitting
# new learning rate range
learning_rate_range = np.arange(0.01, 0.5, 0.05)
fig = plt.figure(figsize=(19, 17))
idx = 1
hyperparameter_score_list = []
# grid search for min_child_weight
for weight in np.arange(0, 4.5, 0.5):
    train = []
    test = []
    for lr in learning_rate_range:
        xgb_classifier = xgb.XGBClassifier(eta=lr, reg_lambda=1, min_child_weight=weight)
        xgb_classifier.fit(x_train, y_train)
        train.append(xgb_classifier.score(x_train, y_train))
        test.append(xgb_classifier.score(x_test, y_test))
        scores = cross_validate(xgb_classifier, x_train, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(scores['test_score'])
        hyperparameter_score_list.append([lr, weight, mean_score])

    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]

    fig.add_subplot(3, 3, idx)
    # spacing = 0.3
    # fig.subplots_adjustst(bottom=spacing)
    idx += 1
    plt.plot(learning_rate_range, train, c=colors[0], label='Training')
    plt.plot(learning_rate_range, test, c='orange', label='Testing')
    plt.xlabel('Learning rate')
    plt.xticks(learning_rate_range)
    plt.ylabel('Accuracy score')
    plt.ylim(0.6, 1)
    plt.legend(prop={'size': 14}, loc=3)
    title = "Min child weight:" + str(weight)
    plt.title(title, size=16)
    plt.ioff()

plt.savefig('./smartforceps_' + classification_task.lower() + '_model'
            '/results/feature_based_model/' + classification_task + '/xgb_classifier_hype_parameter_searching.png')

# for row in hyperparameter_score_list:
df_hyperparameter_score_list = pd.DataFrame(hyperparameter_score_list, columns=["Learning Rate",
                                                                                "Min Child Weight",
                                                                                "Average Accuracy"]).sort_values(
    by=["Average Accuracy", "Learning Rate", "Min Child Weight"], ascending=False
)

df_hyperparameter_score_list.to_csv('./smartforceps_' + classification_task.lower() + '_model'
                                    '/results/feature_based_model/' + classification_task +
                                    '/df_hyperparameter_score_list_xgb_classifier.csv')

# Train with the best model
best_lr = df_hyperparameter_score_list.head(1)["Learning Rate"].to_list()[0]
best_weight = df_hyperparameter_score_list.head(1)["Min Child Weight"].to_list()[0]
xgb_classifier = xgb.XGBClassifier(eta=best_lr, reg_lambda=1, min_child_weight=best_weight)
xgb_classifier.fit(x_train, y_train)
print("Best Model Testing Score: ", xgb_classifier.score(x_test, y_test))

"""
Best Model Testing Score for Task:  0.8105515587529976

Best Model Testing Score for Skill:  0.6551724137931034

"""

# xgboost for feature importance on a classification problem

# train/test split (80/20)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

# XGBoost (different learning rate)
learning_rate_range = np.arange(0.01, 1, 0.05)
test_XG = []
train_XG = []
for lr in learning_rate_range:
    xgb_classifier = xgb.XGBClassifier(eta=lr)
    xgb_classifier.fit(x_train, y_train)
    train_XG.append(xgb_classifier.score(x_train, y_train))
    test_XG.append(xgb_classifier.score(x_test, y_test))
# Line plot
fig = plt.figure(figsize=(18, 10))
plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
plt.plot(learning_rate_range, test_XG, c='m', label='Test')
plt.xlabel('Learning rate')
plt.xticks(learning_rate_range)
plt.ylabel('Accuracy score')
plt.ylim(0.6, 1)
plt.legend(prop={'size': 11}, loc=3)
plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
plt.ioff()
plt.savefig('./smartforceps_' + classification_task.lower() + '_model'
            '/results/feature_based_model/' + classification_task + '/xgb_classifier.png')

# define dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model_xgboost = XGBClassifier()
# fit the model
model_xgboost.fit(x_train, y_train)
# get importance
importance_xgboost = model_xgboost.feature_importances_
# summarize feature importance
for i, v in enumerate(importance_xgboost):
    print('Feature: %0d, Score: %.5f' % (i, v))

df_importance_scores = pd.DataFrame({'Feature': features_list, 'Importance Score': importance_xgboost})
df_importance_scores.sort_values(by=['Importance Score'], ascending=False, inplace=True)

df_importance_scores.to_csv('./smartforceps_' + classification_task.lower() + '_model'
                            '/results/feature_based_model/' + classification_task +
                            '/XGBClassifier_feature_importance.csv')

# plot feature importance
fig = plt.figure(figsize=(24, 10), dpi=320)
spacing = 0.2
fig.subplots_adjust(bottom=spacing)
plt.rcParams.update({'font.size': 14})  # must set in top
grid = plt.GridSpec(4, 4, hspace=0.8, wspace=0.2)
ax = sns.barplot(df_importance_scores['Feature'].to_list(),
                 df_importance_scores['Importance Score'].to_list())
ax.tick_params(axis='x', rotation=45)
ax.axes.set_title('Importance Comparison of Hand Crafted Features for ' + classification_task +
                  ' Classification using XGboost', fontsize=22)
ax.set_xlabel('Feature', fontsize=18)
ax.set_ylabel('Importance Score', fontsize=18)
plt.savefig('./smartforceps_' +
            classification_task.lower() + '_model/'
            'results/feature_based_model/' + classification_task + '/XGBClassifier_feature_importance.png', dpi=400)
plt.show()
