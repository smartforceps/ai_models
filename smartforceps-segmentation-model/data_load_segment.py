# encoding=utf8

"""
    u_net model for time series data segmentation
    
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def feature_normalization(X):
    X = X.astype(float)
    scaler = StandardScaler().fit(X)
    data = scaler.transform(X)
    return data


def load_data(data_name='Smartforceps', subseq=224):
    # X = y = \
    #     X_train_full = X_test_full = y_train_full = y_test_full = \
    #     X_train = X_test = y_train = y_test = \
    #     N_FEATURES = integer_encoded1 = act_classes = class_names = []
    data_output = []
    try:
        if data_name == 'Smartforceps':
            data_output = load_Smartforceps_segment(subseq)
    except:
        print('data not available')

    return data_output


def load_Smartforceps_segment(subseq):
    # df = pd.read_csv('/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/df_force_data_with_label.csv')[
    #     ['LeftCalibratedForceValue',
    #      'RightCalibratedForceValue',
    #      'ForceStatus']]
    df = pd.read_csv('/home/amir/Desktop/smartforceps_ai_models/data/smartforceps_data/df_force_seg_filtered.csv')[
        ['LeftCalibratedForceValue',
         'RightCalibratedForceValue',
         'ForceStatus']]
    df = df.dropna()

    label = ['OFF', 'ON']
    # Show how many training examples exist for each of the two states
    fig = plt.figure(figsize=(6, 8))
    colors = [plt.cm.Set3(i / float(5)) for i in range(5)]
    df['ForceStatus'].value_counts().plot(kind='bar',
                                          title='Data Distribution by Status Class',
                                          color=colors,
                                          rot=45)

    plt.ioff()
    plt.savefig('./results/graphs/class_distribution.png')
    # plt.show()

    np_df = np.array(df.drop('ForceStatus', axis=1))
    norm_np_df = feature_normalization(np_df)
    # norm_np_df = np_df.copy()
    print('first 3 columns of normalized data:')
    print(norm_np_df[:3])

    N_TIME_STEPS = subseq
    N_FEATURES = 2

    step = subseq
    segments = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        ls = norm_np_df[i: i + N_TIME_STEPS, 0]
        rs = norm_np_df[i: i + N_TIME_STEPS, 1]
        segments.append([ls, rs])
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, 1, N_TIME_STEPS, N_FEATURES)
    print('\ndata shape')
    print(reshaped_segments.shape)

    label_vals = df['ForceStatus'].values
    label_encoder = LabelEncoder()
    integer_encoded1 = label_encoder.fit_transform(label_vals)
    integer_encoded2 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded2)
    labels = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        li = onehot_encoded[i: i + N_TIME_STEPS]
        labels.append(li)
    act_classes = len(np.unique(integer_encoded1))
    reshaped_labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1, N_TIME_STEPS, act_classes)
    print('label shape')
    print(reshaped_labels.shape)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        reshaped_segments, reshaped_labels, test_size=0.3, random_state=RANDOM_SEED)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=RANDOM_SEED)
    print('training data shape')
    print(X_train.shape)

    # X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    #     norm_np_df, integer_encoded2, test_size=0.3, random_state=RANDOM_SEED)

    y_train_full = np.reshape(y_train, (y_train.shape[0] * y_train.shape[2], 2))
    unique_y = np.unique(y_train_full)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    # map_y_train = np.zeros((integer_encoded2.shape[0],))
    map_y_train = np.array([class_map[val[0]] for val in y_train_full])

    y_val_full = np.reshape(y_val, (y_val.shape[0] * y_val.shape[2], 2))
    unique_y = np.unique(y_val_full)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    # map_y_val = np.zeros((integer_encoded2.shape[0],))
    map_y_val = np.array([class_map[val[0]] for val in y_val_full])

    y_test_full = np.reshape(y_test, (y_test.shape[0] * y_test.shape[2], 2))
    unique_y = np.unique(y_test_full)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    map_y_test = np.array([class_map[val[0]] for val in y_test_full])

    output = [reshaped_segments,
              reshaped_labels,
              X_train,
              X_val,
              X_test,
              y_train,
              y_val,
              y_test,
              N_FEATURES,
              map_y_train,
              map_y_val,
              map_y_test,
              act_classes,
              list(label)]

    return output
