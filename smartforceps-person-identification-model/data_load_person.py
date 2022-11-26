# encoding=utf8

"""
    model for smartforceps task data classification
    
"""

import numpy as np
import pandas as pd
from random import random
from numpy import stack
from tensorflow.keras.utils import to_categorical
import h5py
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from statistics import mean
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import skew, kurtosis, normaltest
from math import sqrt

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

    df_result['Mean.Force'] = round(mean([df['LeftCalibratedForceValue'].mean(),
                                          df['RightCalibratedForceValue'].mean()]), 4)

    df_result['Max.Force'] = round(max(df['LeftCalibratedForceValue'].max(),
                                       df['RightCalibratedForceValue'].max()), 4)

    df_result['Min.Force'] = round(min(df['LeftCalibratedForceValue'].min(),
                                       df['RightCalibratedForceValue'].min()), 4)

    df_result['Range.Force'] = round(df['LeftCalibratedForceValue'].max() -
                                     df['RightCalibratedForceValue'].min(), 4)

    df_result['Median.Force'] = round(mean([df['LeftCalibratedForceValue'].median(),
                                            df['RightCalibratedForceValue'].median()]), 4)

    df_result['SD.Force'] = round(max(df['LeftCalibratedForceValue'].std(),
                                      df['RightCalibratedForceValue'].std()), 4)

    df_result['Coef.Variance'] = round(max(df[['LeftCalibratedForceValue']].apply(cv)[0],
                                           df[['RightCalibratedForceValue']].apply(cv)[0]), 4)

    df_result['Skewness'] = round(max([abs(skew(df['LeftCalibratedForceValue'])),
                                       abs(skew(df['RightCalibratedForceValue']))]), 4)

    df_result['Kurtosis'] = round(max([abs(kurtosis(df['LeftCalibratedForceValue'])),
                                       abs(kurtosis(df['RightCalibratedForceValue']))]), 4)

    df_result['Normtest'] = round(min(normaltest(df['LeftCalibratedForceValue']).pvalue,
                                      normaltest(df['RightCalibratedForceValue']).pvalue), 4)

    peaks_left, properties_left = find_peaks(bw_filter(df['LeftCalibratedForceValue'].to_numpy()),
                                             distance=5,
                                             prominence=0.5,
                                             wlen=10)
    peaks_right, properties_right = find_peaks(bw_filter(df['RightCalibratedForceValue'].to_numpy()),
                                               distance=5,
                                               prominence=0.5,
                                               wlen=10)

    try:
        df_result['Peaks.Count'] = len(set(peaks_left.tolist() + peaks_right.tolist()))

        df_result['Max.Peak.Value'] = max(max((df['LeftCalibratedForceValue'].iloc[peaks_right] -
                                               df['LeftCalibratedForceValue'].mean()).to_list()),
                                          max((df['RightCalibratedForceValue'].iloc[peaks_right] -
                                               df['RightCalibratedForceValue'].mean()).to_list()))
    except:
        df_result['Peaks.Count'] = 0
        df_result['Max.Peak.Value'] = np.nan

    df_result['Frequency'] = (df_result[['Peaks.Count']].to_numpy()[0][0] + 1) / \
                             df_result[['Duration.Force']].to_numpy()[0][0]

    df_result['Period.Length'] = df_result[['Duration.Force']].to_numpy()[0][0] / (
            df_result[['Peaks.Count']].to_numpy()[0][0] + 1)

    ts_features_left = TsFeatures().transform(TimeSeriesData(df[['MillisecondsSinceRecord',
                                                                 'LeftCalibratedForceValue']].rename(
        columns={'MillisecondsSinceRecord': 'time'})))

    ts_features_right = TsFeatures().transform(TimeSeriesData(df[['MillisecondsSinceRecord',
                                                                  'RightCalibratedForceValue']].rename(
        columns={'MillisecondsSinceRecord': 'time'})))

    df_result['Diff.Force.SD'] = round(max(ts_features_left['std1st_der'],
                                           ts_features_right['std1st_der']), 4)

    df_result['Flat.Spots'] = round(min(ts_features_left['flat_spots'],
                                        ts_features_right['flat_spots']), 4)

    df_result['Trend.Strength'] = round(mean([ts_features_left['trend_strength'],
                                              ts_features_right['trend_strength']]), 4)

    df_result['Linearity'] = round(max([abs(ts_features_left['linearity']),
                                        abs(ts_features_right['linearity'])]), 4)

    df_result['Stability'] = round(mean([ts_features_left['stability'],
                                         ts_features_right['stability']]), 4)

    df_result['Lumpiness'] = round(mean([ts_features_left['lumpiness'],
                                         ts_features_right['lumpiness']]), 4)

    df_result['Crossing.Points'] = round(mean([ts_features_left['crossing_points'],
                                               ts_features_right['crossing_points']]), 4)

    df_result['Entropy'] = round(max(ts_features_left['entropy'],
                                     ts_features_right['entropy']), 4)

    df_result['Heterogeneity'] = round(max(ts_features_left['heterogeneity'],
                                           ts_features_right['heterogeneity']), 4)

    df_result['Spikiness'] = round(max(ts_features_left['spikiness'],
                                       ts_features_right['spikiness']), 4)

    df_result['First.Min.Autocorr'] = round(min(ts_features_left['firstmin_ac'],
                                                ts_features_right['firstmin_ac']), 4)

    df_result['First.Zero.Autocorr'] = round(min(ts_features_left['firstzero_ac'],
                                                 ts_features_right['firstzero_ac']), 4)

    df_result['Autocorr.Function.1'] = round(min(ts_features_left['y_acf1'],
                                                 ts_features_right['y_acf1']), 4)

    df_result['Autocorr.Function.5'] = round(min(ts_features_left['y_acf5'],
                                                 ts_features_right['y_acf5']), 4)

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
            data_output = load_Smartforceps_person(subseq)
    except:
        print('data not available')

    return data_output


def load_Smartforceps_person(window_size):
    # original force data
    # df = pd.read_csv('/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/df_force_data_with_label.csv',
    #                  index_col=0, low_memory=False).iloc[:, [0, 1, 2, 6, 11]]

    ####

    # manual processed set
    # label_manual = ['Dr. Sutherland', 'Candice', 'Michael', 'Abdulrahman']
    label_manual = ['Dr. Sutherland', 'Candice', 'Michael']

    # read force data
    df_manual = pd.read_csv(
        '/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/df_force_data_with_label_2.csv',
        index_col=0,
        low_memory=False
    )
    df_manual = df_manual[df_manual['SurgeonName'].isin(label_manual)][['LeftCalibratedForceValue',
                                                                        'RightCalibratedForceValue',
                                                                        'SurgeonName']]

    ####

    # online processed set
    df_online_in = pd.read_csv(
        '/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/dfProcessed.csv',
    ).sort_values(by=['PartitionKey', 'SegmentNum', 'Time'])[['Timestamp',
                                                              'PartitionKey',
                                                              'SegmentNum',
                                                              'Time',
                                                              'ProngName',
                                                              'Value']]
    df_online_in['PartitionKey'] = df_online_in['PartitionKey'].str.split('case-', 1).str[1]

    df_online_info = pd.read_csv(
        '/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/CaseInfo.csv',
    )[['PartitionKey', 'Surgeon_DisplayName']].rename(columns={'Surgeon_DisplayName': 'SurgeonName'})

    df_online_in['SurgeonName'] = df_online_in['PartitionKey'].map(
        df_online_info.set_index('PartitionKey')['SurgeonName'].to_dict())

    df_online = pd.DataFrame()
    for partition_key in df_online_in['PartitionKey'].unique().tolist():
        df_parted = df_online_in[df_online_in['PartitionKey'] == partition_key]
        for segment_num in df_parted['SegmentNum'].unique().tolist():
            df_select = df_parted[df_parted['SegmentNum'] == segment_num]

            df_pivot = df_select.pivot_table(index='Time',
                                             columns='ProngName',
                                             values='Value').reset_index()

            df_pivot['PartitionKey'] = df_select.head(1)['PartitionKey'].to_list()[0]
            df_pivot['SegmentNum'] = df_select.head(1)['SegmentNum'].to_list()[0]
            df_pivot['SurgeonName'] = df_select.head(1)['SurgeonName'].to_list()[0]
            df_online = df_online.append(df_pivot, ignore_index=True)

    label_online = ['Dr.  T. Surgeon1']

    df_online = df_online[df_online['SurgeonName'].isin(label_online)].rename(
        columns={'LeftForce': 'LeftCalibratedForceValue',
                 'RightForce': 'RightCalibratedForceValue'}
    )[['LeftCalibratedForceValue', 'RightCalibratedForceValue', 'SurgeonName']]

    ####
    # concat both sources of data

    label = label_manual + label_online

    df_concat = pd.concat([df_manual, df_online], axis=0)

    # get minimum class size
    min_class_size = df_concat.shape[0]
    for surgeon in label:
        df_sub = df_concat[df_concat['SurgeonName'] == surgeon]
        if df_sub.shape[0] < min_class_size:
            min_class_size = df_sub.shape[0]

    # cut class size to minimum
    df = pd.DataFrame()
    for surgeon in label:
        df_sub = df_concat[df_concat['SurgeonName'] == surgeon]
        df_sub_cut = df_sub.head(min_class_size + round(random() * df_concat.shape[0] / 1000))
        df = df.append(df_sub_cut, ignore_index=True)

    df.to_csv('/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/df_person_id_data_subsampled.csv')

    ####
    # standaridze data
    data = df[['LeftCalibratedForceValue', 'RightCalibratedForceValue']].copy()
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data
    scaler.fit(data)
    # apply transform
    standardized = scaler.transform(data)
    # inverse transform
    inverse = scaler.inverse_transform(standardized)

    df['LeftCalibratedForceValue'] = inverse[:, 0]
    df['RightCalibratedForceValue'] = inverse[:, 1]

    ####

    # show how many training examples exist for each of the two states
    fig = plt.figure(figsize=(6, 8))
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    df['SurgeonName'].value_counts().plot(kind='bar',
                                          title='Data Distribution by Surgeon Name',
                                          color=colors,
                                          rot=45)
    plt.ioff()
    plt.savefig('./results/graphs/class_distribution.png')
    # plt.show()

    s_labels = {}
    for i in range(len(label)):
        s_labels[label[i]] = i

    df.replace({'SurgeonName': s_labels}, inplace=True)
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
        df_labels.append(int(df.iloc[(window_size * seg): (window_size * (seg + 1))][["SurgeonName"]].median()[0]))

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


#####

def plot_surgeon_data():
    sns.set_theme(style="white")

    window_size = 200

    df = pd.read_csv(
        '/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/df_person_id_data_subsampled.csv',
        index_col=0,
        low_memory=False
    )

    df_segment_full = list()
    segment_features = []
    for seg in range(int(df.shape[0] / window_size)):
        df_seg = df.iloc[(window_size * seg):(window_size * (seg + 1))][['LeftCalibratedForceValue',
                                                                         'RightCalibratedForceValue',
                                                                         'SurgeonName']]

        df_seg['time'] = df_seg.index - df_seg.index[0]

        ts_features_left = TsFeatures().transform(TimeSeriesData(df_seg[['time', 'LeftCalibratedForceValue']]))

        ts_features_right = TsFeatures().transform(TimeSeriesData(df_seg[['time', 'RightCalibratedForceValue']]))

        segment_features.append([round(max((df_seg['LeftCalibratedForceValue'].max() -
                                            df_seg['LeftCalibratedForceValue'].min()),
                                           (df_seg['RightCalibratedForceValue'].max() -
                                            df_seg['RightCalibratedForceValue'].min())), 4),
                                 round(max(df_seg['LeftCalibratedForceValue'].max(),
                                           df_seg['RightCalibratedForceValue'].max()), 4),
                                 round(max(ts_features_left['entropy'],
                                           ts_features_right['entropy']), 4),
                                 df_seg['SurgeonName'].mode()[0]])

        df_segment_full.append(df_seg)

    df_segment_features = pd.DataFrame(segment_features, columns=['Force Range', 'Maximum Force',
                                                                  'Entropy', 'Surgeon Name'])

    df_segment_features.replace({'Dr. Sutherland': 'Surgeon 1', 'Candice': 'Surgeon 2',
                                 'Dr.  T. Surgeon1': 'Surgeon 3', 'Michael': 'Surgeon 4'}, inplace=True)

    df_segment_features.to_csv('/home/amir/Desktop/smartforceps_ai_models/data/'
                               'smartforceps_data/df_person_id_features.csv')

    # calculating force range and force max correlation
    force_range_max_force_corr = df_segment_features['Force Range'].corr(df_segment_features['Maximum Force'])

    # find densities of data points
    densities = []
    for surgeon in df_segment_features['Surgeon Name'].unique().tolist():
        df_sub = df_segment_features[df_segment_features['Surgeon Name'] == surgeon]
        points = df_sub[["Force Range", "Maximum Force"]].to_numpy().tolist()
        total_distance = 0
        count = 0
        i = 0
        for x1, y1 in points:
            for x2, y2 in points[i + 1:]:
                count += 1
                total_distance += sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            i += 1
        densities.append([surgeon, count / total_distance])

    sns.scatterplot(data=df_segment_features, x="Force Range", y="Entropy", hue="Surgeon Name",
                    size="Maximum Force", palette="pastel")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = plt.legend(by_label.values(), by_label.keys(), frameon=False, loc='lower right', fontsize='7')
    legend.texts[0].set_text("")

    plt.ioff()
    plt.savefig('./smartforceps_dl_prediction_models_tf2/smartforceps_person_model/'
                'results/segment_features_scatter.png', dpi=300)

    # get anova for features among surgeons

    df_segment_features = pd.read_csv('/home/amir/Desktop/smartforceps_ai_models/data/'
                                      'smartforceps_data/df_person_id_features.csv', index_col=0)

    # get ANOVA table as R like output
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Ordinary Least Squares (OLS) model
    model = ols('entropy ~ C(surgeon_name)',
                data=df_segment_features.rename(columns={'Force Range': 'force_range',
                                                         'Maximum Force': 'maximum_force',
                                                         'Entropy': 'entropy',
                                                         'Surgeon Name': 'surgeon_name'})).fit()
    anova_table_entropy = sm.stats.anova_lm(model, typ=2)

    model = ols('force_range ~ C(surgeon_name)',
                data=df_segment_features.rename(columns={'Force Range': 'force_range',
                                                         'Maximum Force': 'maximum_force',
                                                         'Entropy': 'entropy',
                                                         'Surgeon Name': 'surgeon_name'})).fit()
    anova_table_force_range = sm.stats.anova_lm(model, typ=2)

    model = ols('maximum_force ~ C(surgeon_name)',
                data=df_segment_features.rename(columns={'Force Range': 'force_range',
                                                         'Maximum Force': 'maximum_force',
                                                         'Entropy': 'entropy',
                                                         'Surgeon Name': 'surgeon_name'})).fit()
    anova_table_maximum_force = sm.stats.anova_lm(model, typ=2)

    from bioinfokit.analys import stat

    res_entropy = stat()
    res_entropy.tukey_hsd(df=df_segment_features.rename(columns={'Force Range': 'force_range',
                                                                 'Maximum Force': 'maximum_force',
                                                                 'Entropy': 'entropy',
                                                                 'Surgeon Name': 'surgeon_name'}),
                          res_var='entropy',
                          xfac_var='surgeon_name',
                          anova_model='entropy ~ C(surgeon_name)')
    res_entropy.tukey_summary

    res_maximum_force = stat()
    res_maximum_force.tukey_hsd(df=df_segment_features.rename(columns={'Force Range': 'force_range',
                                                                       'Maximum Force': 'maximum_force',
                                                                       'Entropy': 'entropy',
                                                                       'Surgeon Name': 'surgeon_name'}),
                                res_var='maximum_force',
                                xfac_var='surgeon_name',
                                anova_model='maximum_force ~ C(surgeon_name)')
    res_maximum_force.tukey_summary

    res_force_range = stat()
    res_force_range.tukey_hsd(df=df_segment_features.rename(columns={'Force Range': 'force_range',
                                                                     'Maximum Force': 'maximum_force',
                                                                     'Entropy': 'entropy',
                                                                     'Surgeon Name': 'surgeon_name'}),
                              res_var='force_range',
                              xfac_var='surgeon_name',
                              anova_model='force_range ~ C(surgeon_name)')
    res_force_range.tukey_summary
