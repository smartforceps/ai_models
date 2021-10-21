import pandas as pd
import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects as ro

# read output data from R
r_file_names = ['SmartForcepsDataRead',
                'SmartForcepsDataProcessed',
                'SmartForcepsDataFeature',
                'SmartForcepsDataFeatureClean']

for file in r_file_names:
    data = pyreadr.read_r('/home/amir/Desktop/smartforceps_dl/data/smartforceps_data/feature data/'
                          + file + '.RData')
    for key, value in data.items():
        vars()[key] = value

csv_file_names = ['SmartForcepsDataProcessed',
                  'SmartForcepsDataFeature']

for file in csv_file_names:
    vars()['df_' + file] = pd.read_csv('/home/amir/Desktop/smartforceps_dl/data/smartforceps_data/feature data/'
                                       + file + '.csv')

del r_file_names, csv_file_names, data, key, value, file

# Re-constructing the time-series features from R script
# defining the R script and loading the instance in Python
r = robjects.r
r['source']('/home/amir/Desktop/smartforceps_dl/data/smartforceps_data/r-feature-extraction.R')
# loading the function we have defined in R.
timeseries_feature_function_r = robjects.globalenv['timeseries_feature']
# converting it into r object for passing into r function
r_case_3_forcedata = ro.conversion.py2rpy(case_3_forcedata.iloc[0:2000])
# invoking the R function and getting the result
df_result_r = timeseries_feature_function_r(r_case_3_forcedata)
# converting it back to a pandas dataframe.
pd_df_results = ro.conversion.rpy2py(df_result_r)

pd_df_results.to_csv('/home/amir/Desktop/smartforceps_dl/data/smartforceps_data/df_hand_features.csv')
