import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from scipy import stats
from statistics import mean
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import skew, kurtosis, normaltest

from sklearn.preprocessing import StandardScaler

from skimage.transform import resize

from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures


def get_min_max_auc_p_value(roc_auc_comb, roc_comb):
    max_key = max(roc_auc_comb, key=roc_auc_comb.get)
    min_key = min(roc_auc_comb, key=roc_auc_comb.get)

    for roc_dict_set in roc_comb:
        if max_key in roc_dict_set.keys():
            max_roc_fpr = roc_dict_set[max_key]['fpr']['macro']
            max_roc_tpr = roc_dict_set[max_key]['tpr']['macro']

        elif min_key in roc_dict_set.keys():
            min_roc_fpr = roc_dict_set[min_key]['fpr']['macro']
            min_roc_tpr = roc_dict_set[min_key]['tpr']['macro']

    min_max_fpr_t_test = stats.ttest_ind(min_roc_fpr, max_roc_fpr)
    min_max_tpr_t_test = stats.ttest_ind(min_roc_tpr, max_roc_tpr)
    min_max_p_value = np.mean([min_max_fpr_t_test.pvalue, min_max_tpr_t_test.pvalue])
    return round(min_max_p_value, 4)


def plot_roc(roc_dict_comb, save_path, difference_p_value, job):
    lw = 2
    plt.figure(figsize=(12, 7))

    n_categories = len(roc_dict_comb)
    colors = [plt.cm.Set3(i / float(n_categories)) for i in range(n_categories)]
    # colors = [plt.cm.Spectral(i / float(n_categories)) for i in range(n_categories)]
    # colors = [plt.cm.tab20b(i / float(n_categories)) for i in range(n_categories)]
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(roc_dict_comb)), colors):
        exp_trial = list(roc_dict_comb[i].keys())[0].split('-')[1]
        exp_trial_model = list(roc_dict_comb[i].keys())[0].split('-')[3]

        if exp_trial == 'no':
            if exp_trial_model == 'lstm':
                trial_name = 'no hand-crafted feature included (LSTM)'
            elif exp_trial_model == 'incept':
                trial_name = 'no hand-crafted feature included (FTFIT)'
        elif exp_trial == 'dash':
            if exp_trial_model == 'lstm':
                trial_name = 'subset 1 hand-crafted features (n = 4) included (LSTM)'
            elif exp_trial_model == 'incept':
                trial_name = 'subset 1 hand-crafted features (n = 4) included (FTFIT)'
        elif exp_trial == 'some':
            if exp_trial_model == 'lstm':
                trial_name = 'subset 2 hand-crafted features (n = 8) included (LSTM)'
            elif exp_trial_model == 'incept':
                trial_name = 'subset 2 hand-crafted features (n = 8) included (FTFIT)'
        elif exp_trial == 'with':
            if exp_trial_model == 'lstm':
                trial_name = 'the full set of hand-crafted features (n = 29) included (LSTM)'
            elif exp_trial_model == 'incept':
                trial_name = 'the full set of hand-crafted features (n = 29) included (FTFIT)'
        elif exp_trial == 'selected':
            if job == 'task':
                if exp_trial_model == 'lstm':
                    trial_name = 'selected features based on importance (n = 7) included (LSTM)'
                elif exp_trial_model == 'incept':
                    trial_name = 'selected features based on importance (n = 7) included (FTFIT)'
            elif job == 'skill':
                if exp_trial_model == 'lstm':
                    trial_name = 'selected features based on importance (n = 9) included (LSTM)'
                elif exp_trial_model == 'incept':
                    trial_name = 'selected features based on importance (n = 9) included (FTFIT)'
        else:
            trial_name = 'unknown set of hand-crafted features included'

        fpr = list(roc_dict_comb[i].values())[0]['fpr']
        tpr = list(roc_dict_comb[i].values())[0]['tpr']
        roc_auc = list(roc_dict_comb[i].values())[0]['roc_auc']
        act_classes = list(roc_dict_comb[i].values())[0]['act_classes']
        plt.plot(fpr['macro'], tpr['macro'], color=color, lw=lw,
                 label='ROC curve for the trial with {0}  (area = {1:0.2f})'
                       ''.format(trial_name, roc_auc['macro']))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.title('Receiver operating characteristic curves', size=24)
    plt.legend(loc="lower right")
    if 0.05 > difference_p_value > 0.0001:
        # plt.figtext(0.36, 0.26,  # task
        #             'The maximum diffenece between trials\nwas significant (p-value = {0})'.format(difference_p_value),
        #             fontsize=13)
        plt.figtext(0.43, 0.32,  # skill
                    'The maximum diffenece between trials\nwas significant (p-value = {0})'.format(difference_p_value),
                    fontsize=13)
    elif difference_p_value < 0.0001:
        # plt.figtext(0.36, 0.26,  # task
        #             'The maximum diffenece between trials\nwas significant (p-value < 0.0001)',
        #             fontsize=13)
        plt.figtext(0.43, 0.32,  # skill
                    'The maximum diffenece between trials\nwas significant (p-value < 0.001)',
                    fontsize=13)

    plt.ioff()
    plt.savefig(save_path + '/results/auc_history_model_compares.pdf')


"""
some-feature: Duration.Force, Range.Force, Coef.Variance, Peaks.Count, Max.Peak.Value, Entropy, Heterogeneity, Spikiness
dash-feature: Duration.Force, Range.Force, Entropy, Spikiness
selected-feayures-task: Entropy, Median Force, Trend Strength, Flat Spots, First Zero Autocorrelation, First Min Autocorrelation, Crossing Points
selected-feayures-skill: Diff Force SD, Min Force, Crossing Points, Normtest, Flat Spots, Linearity, Skewness, First Zero Autocorrelation, Median Force

"""
# for skill model
modeling_for = 'skill'
path = 'ai_models/smartforceps-skill-classification-model/results/smartforceps_skill_model'
# test_experiments = [
# '092921-with-feature-lstm',
# '092921-no-feature-lstm',
# '092921-dash-feature-lstm',
# '092921-with-feature-incept',
# '092921-no-feature-incept',
# '092921-dash-feature-incept'
# ]

# test_experiments = [
#     "100921-dash-feature-incept-subseq200-batch128",
#     "100921-dash-feature-lstm-subseq200-batch128",
#     "100921-no-feature-incept-subseq200-batch128",
#     "100921-no-feature-lstm-subseq200-batch128",
#     "100921-with-feature-incept-subseq200-batch128",
#     "100921-with-feature-lstm-subseq200-batch128",
# ]

test_experiments = ['021523-selected-features-incept-subseq200-batch128-adam',
                    '021523-selected-features-lstm-subseq200-batch128-adam',
                    '021523-dash-features-incept-subseq200-batch128-adam',
                    '021523-dash-features-lstm-subseq200-batch128-adam',
                    '021523-no-feature-incept-subseq200-batch128-adam',
                    '021523-no-feature-lstm-subseq200-batch128-adam']

# for task model
# modeling_for = 'task'
# path = '/home/amir/Desktop/smartforceps_ai_models/smartforceps_dl_prediction_models_tf2/smartforceps_task_model'
# test_experiments = [
# '092621-no-feature-lstm',
# '092421-dash-feature-lstm',
# '092821-no-feature-incept',
# '092721-dash-feature-incept'
# ]

# test_experiments = [
#     "100721-dash-feature-incept-subseq200-batch128",
#     "100821-dash-feature-lstm-subseq200-batch128",
#     "100821-no-feature-incept-subseq200-batch128",
#     "100821-no-feature-lstm-subseq200-batch128",
#     "100821-with-feature-incept-subseq200-batch128",
#     "100821-with-feature-lstm-subseq200-batch128",
# ]

# test_experiments = ['021423-selected-features-incept-subseq200-batch128-adam',
#                     '021423-selected-features-lstm-subseq200-batch128-adam',
#                     '021423-dash-features-incept-subseq200-batch128-adam',
#                     '021423-dash-features-lstm-subseq200-batch128-adam',
#                     '021423-no-feature-incept-subseq200-batch128-adam',
#                     '021423-no-feature-lstm-subseq200-batch128-adam']

roc_dict_list = []
roc_auc_dict = {}
for exp in test_experiments:
    with open(path + '/results/' + exp + '/roc_dicts.pkl', 'rb') as f:
        roc_dict = pickle.load(f)
        roc_dict_list.append({exp: roc_dict})
        roc_auc_dict[exp] = roc_dict['roc_auc']['macro']

min_max_auc_p_value = get_min_max_auc_p_value(roc_auc_dict, roc_dict_list)
plot_roc(roc_dict_list, path, min_max_auc_p_value, modeling_for)

###########################

"""
Create correlation plots between features and classes

"""

window_size = 96

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


def py_tsfeatures(df, duration):
    # df_result = pd.DataFrame(data={'Duration.Force': [round((max(df['MillisecondsSinceRecord']) -
    #                                                          min(df['MillisecondsSinceRecord'])) / 1000, 4)]})
    df_result = pd.DataFrame(data={'Duration.Force': [round(duration, 4)]})

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


os.chdir('..')
df = pd.read_csv(os.getcwd() + '/data/df_force_data_with_label.csv', index_col=0, low_memory=False)
os.chdir('data-analysis')

label_task = ['Coagulation', 'Pulling', 'Manipulation', 'Dissecting', 'Retracting', 'Other']
# df.replace({'TaskCategory': {label_task[0]: 0,
#                              label_task[1]: 1,
#                              label_task[2]: 2,
#                              label_task[3]: 3,
#                              label_task[4]: 4}}, inplace=True)

df.replace({'TaskCategory': {label_task[0]: 0,
                             label_task[1]: 1,
                             label_task[2]: 1,
                             label_task[3]: 1,
                             label_task[4]: 1}}, inplace=True)

label_skill = ['Novice', 'Expert']
df.replace({'SkillClass': {label_skill[0]: 0,
                           label_skill[1]: 1}}, inplace=True)

df = df.dropna()

df_segment = list()
df_labels = list()
feature_label_list = list()
for seg in range(int(df.shape[0] / window_size)):
    df_seg = df.iloc[(window_size * seg):(window_size * (seg + 1))][['LeftCalibratedForceValue',
                                                                     'RightCalibratedForceValue']]

    # add hand crafted features
    seg_duration = df.iloc[(window_size * seg):(window_size * (seg + 1))]['Duration'].mode()[0]
    df_seg['MillisecondsSinceRecord'] = np.linspace(0, 50 * (window_size - 1), window_size)
    pd_df_results = py_tsfeatures(df_seg, seg_duration).replace(np.nan, 0)
    df_seg = df_seg.drop(['MillisecondsSinceRecord'], axis=1)

    df_seg['RresampledFeatures'] = feature_normalization(df_resample(pd_df_results.T,
                                                                     window_size).to_numpy())

    # add hand crafted features

    # df_segment.append(df_seg)
    # df_labels.append(int(df.iloc[(window_size * seg): (window_size * (seg + 1))][["TaskCategory"]].median()[0]))

    pd_df_results['Task.Label'] = int(
        df.iloc[(window_size * seg): (window_size * (seg + 1))][["TaskCategory"]].median()[0])

    pd_df_results['Skill.Label'] = int(
        df.iloc[(window_size * seg): (window_size * (seg + 1))][["SkillClass"]].median()[0])

    feature_label_list.append(pd_df_results.values.tolist()[0])

df_feature_label = pd.DataFrame(feature_label_list, columns=pd_df_results.columns.to_list())

# df_feature_label.replace({'Task.Label': {0: label_task[0],
#                                          1: label_task[1],
#                                          2: label_task[2],
#                                          3: label_task[3],
#                                          4: label_task[4]}}, inplace=True)

df_feature_label.replace({'Task.Label': {0: label_task[0],
                                         1: label_task[5]}}, inplace=True)

df_feature_label.replace({'Skill.Label': {0: label_skill[0],
                                          1: label_skill[1]}}, inplace=True)

os.chdir('..')
df = pd.read_csv(os.getcwd() + '/data/df_feature_label_new.csv')
os.chdir('data-analysis')

######
# read the processed feature data

os.chdir('..')
df_feature_label = pd.read_csv(os.getcwd() + '/data/df_feature_label_new.csv',
                               index_col=0).rename(columns={'Duration.Force': 'Duration Force',
                                                            'Range.Force': 'Range Force',
                                                            'Task.Label': 'Task Label',
                                                            'Skill.Label': 'Skill Label'})
os.chdir('data-analysis')

features_list = ['Duration Force',
                 'Range Force',
                 'Entropy',
                 'Heterogeneity']
# 'Stability']

category = 'Task Label'
# category = 'Skill Label'

df_feature_subset = df_feature_label[features_list]

feature_subset_norm = feature_normalization(
    df_feature_subset[~df_feature_subset.isin([np.nan, np.inf, -np.inf]).any(1)])

df_feature_subset_norm = pd.DataFrame(data=feature_subset_norm, columns=features_list)
df_feature_subset_norm[category] = df_feature_label[category].tolist()

# remove outlier rows
df_feature_subset_out_removed = df_feature_subset_norm[(np.abs(stats.zscore(df_feature_subset_norm.loc[:,
                                                                            df_feature_subset_norm.columns !=
                                                                            category])) < 3).all(axis=1)]

os.chdir('..')
path = os.getcwd() + '/data/feature data/'
os.chdir('data-analysis')

sns.pairplot(df_feature_subset_out_removed, kind="reg", hue=category, corner=True,
             plot_kws={'scatter_kws': {'alpha': 0.2}})

plt.ioff()
plt.savefig(path + 'features_pairwise_plot_' + category + '_with_' + features_list[-1] + '_new.pdf')
