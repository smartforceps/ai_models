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


def py_tsfeatures(df):
    df_result = pd.DataFrame(data={'Duration.Force': [round((max(df['MillisecondsSinceRecord']) -
                                                             min(df['MillisecondsSinceRecord'])) / 1000, 4)]})

    # df_result['Mean.Force'] = round(mean([df['LeftCalibratedForceValue'].mean(),
    #                                       df['RightCalibratedForceValue'].mean()]), 4)
    #
    # df_result['Max.Force'] = round(max(df['LeftCalibratedForceValue'].max(),
    #                                    df['RightCalibratedForceValue'].max()), 4)
    #
    # df_result['Min.Force'] = round(min(df['LeftCalibratedForceValue'].min(),
    #                                    df['RightCalibratedForceValue'].min()), 4)

    df_result['Range.Force'] = round(df['LeftCalibratedForceValue'].max() -
                                     df['RightCalibratedForceValue'].min(), 4)

    # df_result['Median.Force'] = round(mean([df['LeftCalibratedForceValue'].median(),
    #                                         df['RightCalibratedForceValue'].median()]), 4)
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

    ts_features_left = TsFeatures().transform(TimeSeriesData(df[['MillisecondsSinceRecord',
                                                                 'LeftCalibratedForceValue']].rename(
        columns={'MillisecondsSinceRecord': 'time'})))

    ts_features_right = TsFeatures().transform(TimeSeriesData(df[['MillisecondsSinceRecord',
                                                                  'RightCalibratedForceValue']].rename(
        columns={'MillisecondsSinceRecord': 'time'})))

    # df_result['Diff.Force.SD'] = round(max(ts_features_left['std1st_der'],
    #                                        ts_features_right['std1st_der']), 4)
    #
    # df_result['Flat.Spots'] = round(min(ts_features_left['flat_spots'],
    #                                     ts_features_right['flat_spots']), 4)
    #
    # df_result['Trend.Strength'] = round(mean([ts_features_left['trend_strength'],
    #                                           ts_features_right['trend_strength']]), 4)
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
    # df_result['Crossing.Points'] = round(mean([ts_features_left['crossing_points'],
    #                                            ts_features_right['crossing_points']]), 4)

    df_result['Entropy'] = round(max(ts_features_left['entropy'],
                                     ts_features_right['entropy']), 4)

    df_result['Heterogeneity'] = round(max(ts_features_left['heterogeneity'],
                                           ts_features_right['heterogeneity']), 4)

    # df_result['Spikiness'] = round(max(ts_features_left['spikiness'],
    #                                    ts_features_right['spikiness']), 4)
    #
    # df_result['First.Min.Autocorr'] = round(min(ts_features_left['firstmin_ac'],
    #                                             ts_features_right['firstmin_ac']), 4)
    #
    # df_result['First.Zero.Autocorr'] = round(min(ts_features_left['firstzero_ac'],
    #                                              ts_features_right['firstzero_ac']), 4)
    #
    # df_result['Autocorr.Function.1'] = round(min(ts_features_left['y_acf1'],
    #                                              ts_features_right['y_acf1']), 4)
    #
    # df_result['Autocorr.Function.5'] = round(min(ts_features_left['y_acf5'],
    #                                              ts_features_right['y_acf5']), 4)

    return df_result


def feature_normalization(X):
    X = X.astype(float)
    scaler = StandardScaler().fit(X)
    data = scaler.transform(X)
    return data


def one_hot_encode_and_weight(data):
    unique_y = np.unique(data)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    # one-hot encode
    map_y = np.zeros((data.shape[0],))
    map_y = np.array([class_map[val] for val in data])
    categorical_y = to_categorical(map_y)

    # calculating the class weights
    y_count = Counter(map_y)
    tablew = np.zeros((len(unique_y),))
    for i in range(len(unique_y)):
        tablew[i] = y_count[i] / map_y.shape[0]

    return [categorical_y, map_y, tablew]


def df_resample(df1, num=1):
    df2 = pd.DataFrame()
    for key, value in df1.iteritems():
        temp = value.to_numpy() / value.abs().max()  # normalize
        resampled = resize(temp, (num, 1), mode='edge') * value.abs().max()  # de-normalize
        df2[key] = resampled.flatten().round(2)
    return df2


def load_data(data_name='Smartforceps', subseq=224):
    data_output = []
    try:
        if data_name == 'Smartforceps':
            data_output = load_Smartforceps_task(subseq)
    except:
        print('data not available')

    return data_output


def load_Smartforceps_task(window_size):
    os.chdir('..')
    # augmented force data
    df = pd.read_csv(
        './data/df_force_data_with_label_aug.csv',
        index_col=0, low_memory=False)

    label = ['Coagulation', 'Pulling', 'Manipulation', 'Dissecting', 'Retracting']

    # show how many training examples exist for each of the two states
    fig = plt.figure(figsize=(6, 8))
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    df['TaskCategory'].value_counts().plot(kind='bar',
                                           title='Data Distribution by Task Category',
                                           color=colors,
                                           rot=45)
    plt.ioff()
    plt.savefig('./results/graphs/class_distribution.png')
    # plt.show()

    df.replace({'TaskCategory': {label[0]: 0,
                                 label[1]: 1,
                                 label[2]: 2,
                                 label[3]: 3,
                                 label[4]: 4}}, inplace=True)
    df = df.dropna()

    df_segment = list()
    df_labels = list()
    for seg in range(int(df.shape[0] / window_size)):
        df_seg = df.iloc[(window_size * seg):(window_size * (seg + 1))][['LeftCalibratedForceValue',
                                                                         'RightCalibratedForceValue']]

        # add hand crafted features
        # df_seg['MillisecondsSinceRecord'] = np.linspace(0, 50 * (window_size - 1), window_size)
        # pd_df_results = py_tsfeatures(df_seg).replace(np.nan, 0)
        # df_seg = df_seg.drop(['MillisecondsSinceRecord'], axis=1)
        #
        # df_seg['RresampledFeatures'] = feature_normalization(df_resample(pd_df_results.T,
        #                                                                  window_size).to_numpy())
        # add hand crafted features

        df_segment.append(df_seg)
        df_labels.append(int(df.iloc[(window_size * seg): (window_size * (seg + 1))][["TaskCategory"]].median()[0]))

    X = stack(df_segment)
    y = stack(df_labels)

    N_FEATURES = X.shape[2]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                      y_train_val,
                                                      test_size=0.3,
                                                      random_state=42)

    # one hot encode y
    [categorical_y_train, map_y_train, tablew_train] = one_hot_encode_and_weight(y_train)
    [categorical_y_val, map_y_val, tablew_val] = one_hot_encode_and_weight(y_val)
    [categorical_y_test, map_y_test, tablew_test] = one_hot_encode_and_weight(y_test)

    output = [X,
              y,
              X_train,
              X_val,
              X_test,
              categorical_y_train,
              categorical_y_val,
              categorical_y_test,
              map_y_train,
              tablew_train,
              map_y_test,
              N_FEATURES,
              y_train,
              y_val,
              len(label),
              list(label)]

    return output
