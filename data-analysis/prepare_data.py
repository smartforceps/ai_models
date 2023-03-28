import os
import pandas as pd
import numpy as np
import warnings
import datetime
from skimage.transform import resize
from bisect import bisect_left as lower_bound

warnings.filterwarnings('ignore')

os.chdir('..')
force_data_labels = pd.read_csv(os.getcwd() + '/data/Data Info.csv')
os.chdir('data-analysis')


def hhmmss_to_sec(hhmmss):
    date_time = datetime.datetime.strptime(hhmmss, "%H:%M:%S")
    a_timedelta = date_time - datetime.datetime(1900, 1, 1)
    return int(a_timedelta.total_seconds())


def df_resample(df1, num=1):
    df2 = pd.DataFrame()
    for key, value in df1.iteritems():
        temp = value.to_numpy() / value.abs().max()  # normalize
        resampled = resize(temp, (num, 1), mode='edge') * value.abs().max()  # de-normalize
        df2[key] = resampled.flatten().round(2)
    return df2


def read_data(start_case, end_case):
    df_data = pd.DataFrame(columns=['Case',
                                    'DataSection',
                                    'SampleNumber',
                                    'MillisecondsSincePowerUp',
                                    'LeftRawVoltageValue',
                                    'RightRawVoltageValue',
                                    'LeftCalibratedForceValue',
                                    'RightCalibratedForceValue'])

    for case in range(start_case, end_case + 1):
        print('processing case # ' + str(case))
        for subdir, dirs, files in os.walk(r'./log data' +
                                           '/Case ' +
                                           str(case)):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".txt"):
                    df_read = pd.read_table(filepath,
                                            delim_whitespace=True,
                                            skiprows=1,
                                            names=('SampleNumber',
                                                   'MillisecondsSincePowerUp',
                                                   'LeftRawVoltageValue',
                                                   'RightRawVoltageValue',
                                                   'LeftCalibratedForceValue',
                                                   'RightCalibratedForceValue')).assign(
                        Case=str(case),
                        DataSection=filename.split('part-')[1][0])
                    df_read['MillisecondsSinceRecord'] = df_read['MillisecondsSincePowerUp'].apply(
                        lambda x: x - df_read['MillisecondsSincePowerUp'].iloc[0])
                    df_data = df_data.append(df_read)

    df_data.to_csv('df_force_data.csv')
    print('import complete')

    return df_data


def assign_labels(df_data, data_labels):
    data_labels['TimeStartSec'] = data_labels['TimeStart'].apply(lambda x: hhmmss_to_sec(x))
    data_labels['TimeEndSec'] = data_labels['TimeEnd'].apply(lambda x: hhmmss_to_sec(x))
    data_labels['SectionNum'] = data_labels['Remarks'].apply(lambda x: int(x.split('Section ')[1][0]))
    data_labels['TimeRange'] = data_labels[['TimeStartSec', 'TimeEndSec']].values.tolist()

    df_data_with_label = df_data.copy()
    df_data_with_label['Duration'] = 'NA'
    df_data_with_label['ForceStatus'] = 'OFF'
    df_data_with_label['SkillClass'] = 'NA'
    df_data_with_label['TaskCategory'] = 'NA'
    df_data_with_label['SurgeonName'] = 'NA'

    counter = 0
    for idx, row in data_labels.iterrows():
        counter += 1
        if counter % 100 == 0:
            print('processing force record # ' + str(counter))

        time_range = np.arange(row['TimeStartSec'] * 1000, row['TimeEndSec'] * 1000 + 50, 50).tolist()
        case_num = row['Case']
        section_num = row['SectionNum']

        df_data_with_label.loc[((df_data_with_label['Case'] == case_num) &
                                (df_data_with_label['DataSection'] == section_num) &
                                (df_data_with_label['MillisecondsSinceRecord'].isin(time_range))),
                               'ForceStatus'] = 'ON'

        if len(row['Remarks'].split('-')) == 1:
            user_skill_check = 'Expert'
        else:
            user_skill_check = 'Novice'

        if len(row['Remarks'].split('-')) == 1:
            user_name_check = 'Dr. Sutherland'
        else:
            try:
                user_name_check = row['Remarks'].split(' - ')[1].split('By ')[1]
            except:
                user_name_check = 'Dr. Sutherland'

        df_data_with_label.loc[((df_data_with_label['Case'] == case_num) &
                                (df_data_with_label['DataSection'] == section_num) &
                                (df_data_with_label['MillisecondsSinceRecord'].isin(time_range))),
                               'Duration'] = row['TimeEndSec'] - row['TimeStartSec']

        df_data_with_label.loc[((df_data_with_label['Case'] == case_num) &
                                (df_data_with_label['DataSection'] == section_num) &
                                (df_data_with_label['MillisecondsSinceRecord'].isin(time_range))),
                               'SkillClass'] = user_skill_check

        df_data_with_label.loc[((df_data_with_label['Case'] == case_num) &
                                (df_data_with_label['DataSection'] == section_num) &
                                (df_data_with_label['MillisecondsSinceRecord'].isin(time_range))),
                               'TaskCategory'] = row['Task'].split(' ')[0]

        df_data_with_label.loc[((df_data_with_label['Case'] == case_num) &
                                (df_data_with_label['DataSection'] == section_num) &
                                (df_data_with_label['MillisecondsSinceRecord'].isin(time_range))),
                               'SurgeonName'] = user_name_check

    df_data_with_label.to_csv('df_force_data_with_label.csv')

    return df_data_with_label


def apply_rule_based_segmentation(df):
    df = df.reset_index().replace({'ForceStatus': {'ON': 1, 'OFF': 0}})
    df['RightCalibratedForceMA'] = df['RightCalibratedForceValue'].rolling(window=5).mean()
    df['LeftCalibratedForceMA'] = df['LeftCalibratedForceValue'].rolling(window=5).mean()
    df['MASegment'] = df.apply(lambda x: 0 if (abs(x['RightCalibratedForceMA']) <= 0.3 and
                                               abs(x['LeftCalibratedForceMA']) <= 0.3) else 1, axis=1)
    df['MASegmentChange'] = df['MASegment'].diff()

    # pad_length = 10
    # for idx, row in df.iterrows():
    #     if (row['MASegmentChange'] == -1) and (idx < df.shape[0] - pad_length):
    #         df.loc[idx:idx + pad_length, 'MASegment'] = 1
    #     elif (row['MASegmentChange'] == -1) and (idx > df.shape[0] - pad_length):
    #         df.loc[idx:df.shape[0], 'MASegment'] = 1
    #     if (row['MASegmentChange'] == 1) and (idx > pad_length):
    #         df.loc[idx - pad_length:idx, 'MASegment'] = 1
    #     elif (row['MASegmentChange'] == 1) and (idx < pad_length):
    #         df.loc[0:idx, 'MASegment'] = 1

    indexNames = df[(df['MASegment'] == 0) & (df['ForceStatus'] == 0)].index
    df_select = df.drop(indexNames.to_numpy().tolist())
    df_select = df_select.replace({'ForceStatus': {1: 'ON', 0: 'OFF'}})

    df_select.to_csv('df_force_seg_filtered.csv')

    return df_select


def prepare_augmented_seg_data():
    resample_length = 200

    try:
        os.chdir('..')
        df_segs_list = pd.read_csv(os.getcwd() + '/data/df_segs_list.csv', index_col=0)
        os.chdir('data-analysis')
    except:
        os.chdir('..')
        df_force_seg_processed = pd.read_csv(os.getcwd() + '/data/SmartForcepsDataProcessed.csv', index_col=0)
        os.chdir('data-analysis')
        df_force_seg_processed.replace({'TaskType': {'Pulling': 0,
                                                     'Manipulation': 1,
                                                     'Dissecting': 2,
                                                     'Retracting': 3,
                                                     'Coagulation': 4}}, inplace=True)

        segs_list = []
        for segment_num in df_force_seg_processed['SegmentNumOverall'].unique():
            if segment_num % 100 == 0:
                print('processing force segment # ' + str(segment_num))

            df_seg_left = df_force_seg_processed[(df_force_seg_processed['ProngName'] == 'LeftForce') &
                                                 (df_force_seg_processed['SegmentNumOverall'] == segment_num)]
            df_seg_right = df_force_seg_processed[(df_force_seg_processed['ProngName'] == 'RightForce') &
                                                  (df_force_seg_processed['SegmentNumOverall'] == segment_num)]
            segs_list.append([segment_num, pd.concat([df_seg_left, df_seg_right])['TaskType'].mode()[0]] +
                             df_resample(df_seg_left[['Value']], resample_length)['Value'].tolist() +
                             df_resample(df_seg_right[['Value']], resample_length)['Value'].tolist())

        df_segs_list = pd.DataFrame(segs_list,
                                    columns=['SegmentNumOverall',
                                             'TaskType'] + [str(x + 1) for x in range(2 * resample_length)])

        os.chdir('..')
        df_segs_list.to_csv(os.getcwd() + '/data/df_segs_list.csv')
        os.chdir('data-analysis')

    # get sample tests
    df_segs_list_model_test = df_segs_list.sample(frac=0.2, random_state=resample_length)
    df_segs_list = df_segs_list.drop(df_segs_list_model_test.index)

    seg_num = []
    task_type = []
    left_force = []
    right_force = []
    count = 1
    for row in df_segs_list_model_test.iterrows():
        seg_num = seg_num + [count] * resample_length
        task_type = task_type + [row[1]['TaskType']] * resample_length
        left_force = left_force + row[1].to_list()[2:resample_length + 2]
        right_force = right_force + row[1].to_list()[resample_length + 2:]
        count += 1

    df_model_test = pd.DataFrame([seg_num, left_force, right_force, task_type]).transpose()
    df_model_test.columns = ['SegmentNumber', 'LeftCalibratedForceValue', 'RightCalibratedForceValue', 'TaskCategory']

    os.chdir('..')
    df_model_test.to_csv(os.getcwd() + '/data/df_force_data_with_label_model_test.csv')
    os.chdir('data-analysis')

    df_segs_list_no_coag = df_segs_list[df_segs_list['TaskType'] != 4]
    df_segs_list_coag = df_segs_list[df_segs_list['TaskType'] == 4]

    # save the left and right prong data as train and test txt files
    df_segs_list_train = df_segs_list_no_coag.sample(frac=0.7, random_state=resample_length)
    df_segs_list_test = df_segs_list_no_coag.drop(df_segs_list_train.index)

    os.chdir('DTW_data_augmentation/')
    df_segs_list_train.drop(columns=['SegmentNumOverall']).to_csv(
        r'SmartForceps_Archive/TaskSegmentsNoCoagOnlyTrain/TaskSegmentsNoCoagOnlyTrain_TRAIN',
        header=None,
        index=None,
        sep=',',
        mode='a')

    df_segs_list_test.drop(columns=['SegmentNumOverall']).to_csv(
        r'SmartForceps_Archive/TaskSegmentsNoCoagOnlyTrain/TaskSegmentsNoCoagOnlyTrain_TEST',
        header=None,
        index=None,
        sep=',',
        mode='a')

    # run DTW data augmentation method for left and right prongs
    os.system('python3 spawn.py --datasetname=TaskSegmentsNoCoagOnlyTrain --n_reps=10 '
              '--n_base=2 --k=1 --ssg_epochs=1 --input_suffix=_TRAIN --output_suffix=_EXP_TRAIN')

    # read data augmented and convert back to dataframe
    if os.path.exists('SmartForceps_Archive/TaskSegmentsNoCoagOnlyTrain/TaskSegmentsNoCoagOnlyTrain_EXP_TRAIN'):
        f = open('SmartForceps_Archive/TaskSegmentsNoCoagOnlyTrain/TaskSegmentsNoCoagOnlyTrain_EXP_TRAIN', "rt")
        text = f.readlines()
        out_list = [[float(ln) for ln in ls.split(',')] for ls in text]

        df_segs_aug = pd.DataFrame(out_list,
                                   columns=['TaskType'] + [str(x + 1) for x in range(2 * resample_length)])

        df_segs_list_coag = df_segs_list_coag.drop(columns=['SegmentNumOverall'])
        df_segs_list_coag['TaskType'] = 4.0

        df_segs_aug = df_segs_aug.append(df_segs_list_coag, ignore_index=True)

        df_segs_aug.replace({'TaskType': {0: 1,
                                          1: 2,
                                          2: 3,
                                          3: 4,
                                          4: 0}}, inplace=True)

        df_segs_aug.replace({'TaskType': {0: 'Coagulation',
                                          1: 'Pulling',
                                          2: 'Manipulation',
                                          3: 'Dissecting',
                                          4: 'Retracting'}}, inplace=True)

        seg_num = []
        task_type = []
        left_force = []
        right_force = []
        count = 1
        for row in df_segs_aug.iterrows():
            seg_num = seg_num + [count] * resample_length
            task_type = task_type + [row[1]['TaskType']] * resample_length
            left_force = left_force + row[1].to_list()[1:resample_length + 1]
            right_force = right_force + row[1].to_list()[resample_length + 1:]
            count += 1

        df = pd.DataFrame([seg_num, left_force, right_force, task_type]).transpose()
        df.columns = ['SegmentNumber', 'LeftCalibratedForceValue', 'RightCalibratedForceValue', 'TaskCategory']

        os.chdir('..')
        df.to_csv(os.getcwd() + '/data/df_force_data_with_label_aug_only_train.csv')
        os.chdir('data-analysis')
    else:
        df_segs_aug = 'augmented data not available'

    return df_segs_aug


def prepare_balanced_task_data():
    resample_length = 200

    try:
        os.chdir('..')
        df_segs_list = pd.read_csv(os.getcwd() + '/data/df_segs_list.csv', index_col=0)
        os.chdir('data-analysis')
    except:
        os.chdir('..')
        df_force_seg_processed = pd.read_csv(os.getcwd() + '/data/SmartForcepsDataProcessed.csv', index_col=0)
        os.chdir('data-analysis')
        df_force_seg_processed.replace({'TaskType': {'Pulling': 0,
                                                     'Manipulation': 1,
                                                     'Dissecting': 2,
                                                     'Retracting': 3,
                                                     'Coagulation': 4}}, inplace=True)

        segs_list = []
        for segment_num in df_force_seg_processed['SegmentNumOverall'].unique():
            if segment_num % 100 == 0:
                print('processing force segment # ' + str(segment_num))

            df_seg_left = df_force_seg_processed[(df_force_seg_processed['ProngName'] == 'LeftForce') &
                                                 (df_force_seg_processed['SegmentNumOverall'] == segment_num)]
            df_seg_right = df_force_seg_processed[(df_force_seg_processed['ProngName'] == 'RightForce') &
                                                  (df_force_seg_processed['SegmentNumOverall'] == segment_num)]
            segs_list.append([segment_num, pd.concat([df_seg_left, df_seg_right])['TaskType'].mode()[0]] +
                             df_resample(df_seg_left[['Value']], resample_length)['Value'].tolist() +
                             df_resample(df_seg_right[['Value']], resample_length)['Value'].tolist())

        df_segs_list = pd.DataFrame(segs_list,
                                    columns=['SegmentNumOverall',
                                             'TaskType'] + [str(x + 1) for x in range(2 * resample_length)])

        os.chdir('..')
        df_segs_list.to_csv(os.getcwd() + '/data/df_segs_list.csv')
        os.chdir('data-analysis')

    df_segs_list_no_coag = df_segs_list[df_segs_list['TaskType'] != 4]
    df_segs_list_coag = df_segs_list[df_segs_list['TaskType'] == 4]

    df_segs_list_sub_coag = df_segs_list_coag.sample(frac=0.5, random_state=resample_length)

    df_segs_balanced = df_segs_list_no_coag.append(df_segs_list_sub_coag, ignore_index=True)

    df_segs_balanced.replace({'TaskType': {0: 1,
                                           1: 2,
                                           2: 3,
                                           3: 4,
                                           4: 0}}, inplace=True)

    df_segs_balanced.replace({'TaskType': {0: 'Coagulation',
                                           1: 'Pulling',
                                           2: 'Manipulation',
                                           3: 'Dissecting',
                                           4: 'Retracting'}}, inplace=True)

    """
        Coagulation     1170
        Manipulation     323
        Pulling          316
        Retracting       149
        Dissecting       127
        
    """

    seg_num = []
    task_type = []
    left_force = []
    right_force = []
    count = 1
    for row in df_segs_balanced.iterrows():
        seg_num = seg_num + [count] * resample_length
        task_type = task_type + [row[1]['TaskType']] * resample_length
        left_force = left_force + row[1].to_list()[2:resample_length + 2]
        right_force = right_force + row[1].to_list()[resample_length + 2:]
        count += 1

    df = pd.DataFrame([seg_num, left_force, right_force, task_type]).transpose()
    df.columns = ['SegmentNumber', 'LeftCalibratedForceValue', 'RightCalibratedForceValue', 'TaskCategory']

    os.chdir('..')
    df.to_csv(os.getcwd() + '/data/df_force_data_with_label_balanced.csv')
    os.chdir('data-analysis')


# read force data and assign labels
if os.path.isfile('df_force_seg_filtered.csv'):
    print("Filtered labeled data file exist")
    # read data
    df_force_seg_filtered = pd.read_csv('df_force_seg_filtered.csv', index_col=0)

elif os.path.isfile('df_force_data_with_label.csv'):
    print("Unfiltered labeled data file exist")
    # read data
    df_force_data_with_skill_label = pd.read_csv('df_force_data_with_label.csv', index_col=0)
    df_force_seg_filtered = apply_rule_based_segmentation(df_force_data_with_skill_label)

elif os.path.isfile('df_force_data.csv'):
    print("Force data file exist")
    # read data
    df_force_data = pd.read_csv('df_force_data.csv', index_col=0)
    df_force_data_with_label = assign_labels(df_force_data, force_data_labels)
    df_force_seg_filtered = apply_rule_based_segmentation(df_force_data_with_label)

elif os.path.isfile('df_force_data_with_label_aug.csv'):
    print("Augmented force data file exist")
    # read data
    df_segs_list_aug = pd.read_csv('df_force_data_with_label_aug.csv', index_col=0)

else:
    print("Force data file not exist")
    # import data
    case_range = [2, 51]
    df_force_data = read_data(case_range[0], case_range[1])
    df_force_data_with_label = assign_labels(df_force_data, force_data_labels)
    df_force_seg_filtered = apply_rule_based_segmentation(df_force_data_with_label)
    df_segs_list_aug = prepare_augmented_seg_data()
