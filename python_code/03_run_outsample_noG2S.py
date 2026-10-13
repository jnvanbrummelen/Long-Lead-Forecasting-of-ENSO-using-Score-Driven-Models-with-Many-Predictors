#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
""" 
@author: Janneke van Brummelen
"""
from python_code.methods.Outsample_estimation import *

####################
##### Settings #####
####################
output_folder = "../output/intermediary_data/outsample/"
input_folder_name = "../data"
data_file_name = "Book1982_2025_autosim_HIGHRES_OISST"

insample_length = 276
skip_last = 0
skip_first = 0

bool_use_dates = True
bool_use_random_search = True
bool_use_pre_estim = False
horizons = [1, 3, 6, 12, 14, 16, 18]

burn = 6
num_avg = 6
sig_level = 0.01

#####################
### Data settings ###
#####################
date_name = "DATE"
Y_label = 'EN3.4'
X_labels = ['RB', 'WPAC', 'WPAC2', 'WPAC3', 'WPAC4', '50fin',
       '100cold', '100fin1', '100fin2', '150fin1', '150fin2', '200fin1',
       '200fin2', '250fin1', '250fin2', '300fin1', '300fin2', '400fin1',
       '400fin2', '500fin1', '500fin2', 'wnd160.200_0.10', 'wnd180.220_-4.4', 'wnd180.210_-10.0']

# build lag lists
max_lag_UB = 36   
X_min_lags = [6,  1, 1, 1, 1, 1,  6,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1, 1,  1,  1,  1,  6,  6,  6]
X_max_lags = [11, 5, 5, 5, 5, 5, 11, 11, 11, 11, 18, 11, 5, 18, 18, 11, 18, 5, 18, 11, 18, 11, 11, 18]
X_max_lags = [max(x_max, x_min - 1) for x_max, x_min in zip(X_max_lags, X_min_lags)]
 
X_lags = []
for i, lab in enumerate(X_labels):
    lags = [j for j in range(X_min_lags[i],X_max_lags[i]+1)]
    if lab in ['WPAC','WPAC2','WPAC3' ,'WPAC4','50fin1','400fin1', '200fin2']:
        lags = lags + [j for j in range(12,19)]
    if lab in ['50fin', '100fin1', '150fin1', '200fin1', '250fin1', '300fin1', '400fin11', '500fin1', '100fin2', '150fin2', '200fin2', '250 fin2', '300fin2', '100cold', 'wnd160.200_0.10', 'wnd180.220_-4.4', 'wnd180.210_-10.0']:
        lags = lags + [j for j in range(19,37)] 
    X_lags = X_lags + [np.array(lags)]

################### Reading and transforming data #######################
dfData = pd.read_excel(input_folder_name+"/"+data_file_name+".xlsx",na_values='.', header=0)
lTotLabels = [Y_label] + X_labels

data_matrix = dfData[lTotLabels].values
Y_mat = data_matrix.astype(float) 
N = Y_mat.shape[1] 
string_dates = dfData[date_name].values
 
T_tot = Y_mat.shape[0]
if skip_first > 0:
    Y_mat = Y_mat[skip_first:,:] 
    string_dates = string_dates[skip_first:]
if skip_last > 0:
    Y_mat = Y_mat[:-skip_last,:] 
    string_dates = string_dates[:-skip_last] 

if bool_use_dates:
    num_dates, string_dates = StringToDate(string_dates)
 
Y_vec = Y_mat[:,0]
T = Y_vec.shape[0]

monthly_means = np.array([np.mean(Y_vec[i::12]) for i in range(12)])
repeated_means = np.tile(monthly_means, int(np.ceil(T / 12)))
Y_vec_anom = Y_vec.reshape(-1, 1) - repeated_means[:Y_vec.shape[0]].reshape(-1, 1)

#########################################################################
######################### CONFIGURATIONS ################################
#########################################################################

configurations = []

##########################################################################
# No exogenous variables
##########################################################################

for num_comps, setting in [
    (0, 'A'),
    (3, 'B'),
    (3, 'C'),
    (5, 'B'),
    (5, 'C'),
]:
    for bool_use_seas_dummies in [False, True]:
        if bool_use_seas_dummies and (num_comps == 0 or setting == 'C'):
            continue
        configurations.append({
            "include_X": False,
            "method": "",
            "num_comps": num_comps,
            "ar_lags": 0,
            "setting": setting,
            "bool_use_seas_dummies": bool_use_seas_dummies,
            "bool_use_PCA": False,   # irrelevant if include_X=False
        })

##########################################################################
# With only exogenous variables Lasso
##########################################################################

for bool_use_PCA in [False, True]:
    configurations.append({
        "include_X": True,
        "method": 'Lasso',
        "num_comps": 0,
        "ar_lags": 0,
        "setting": 'A',
        "bool_use_seas_dummies": False,
        "bool_use_PCA": bool_use_PCA,
    })

##########################################################################
# AR models
##########################################################################
for ar_lag in [1,3]:
    configurations.append({
                "include_X": False,
                "method": '',
                "num_comps": 0,
                "ar_lags": ar_lag,
                "setting": 'A',
                "bool_use_seas_dummies": False,
                "bool_use_PCA": False,
            })

#########################################################################
#########################################################################
########################### ESTIMATION ##################################
#########################################################################
#########################################################################
MSE_mat = np.zeros((len(configurations),len(horizons)))

config_names = []
def run_forecasting_configs(configurations):
    for i, config in enumerate(configurations):
        include_X = config["include_X"]
        method = config["method"]
        num_comps = config["num_comps"]
        setting = config["setting"]
        ar_lags = config["ar_lags"]
        bool_use_seas_dummies = config["bool_use_seas_dummies"]
        bool_use_PCA = config["bool_use_PCA"]

        if ar_lags > 0:
            lYlags = [i for i in range(1, ar_lags + 1)]
            X_lags_ast = [np.array(lYlags)] + X_lags
        else:
            X_lags_ast = X_lags

        bool_use_anomalies = (ar_lags > 0)
        if bool_use_anomalies:
            Y_vec_ast = Y_vec_anom
        else:
            Y_vec_ast = Y_vec

        if include_X:
            X_vars = Y_mat[:, 1:]
        else:
            X_vars = np.zeros((Y_vec.shape[0], 0))

        for h, horizon in enumerate(horizons):
            if horizon > 1 and not include_X:
                bool_use_pre_estim_ast = True
            else:
                bool_use_pre_estim_ast = bool_use_pre_estim
            if include_X and num_comps > 0:
                bool_use_pre_estim_noexog = True
            else:
                bool_use_pre_estim_noexog = False
            mse, config_name, _ = EstimationAndForecast(output_folder, insample_length, Y_vec_ast, X_vars, num_dates, string_dates, X_lags_ast, X_labels,
                                  bool_use_random_search, bool_use_dates, sig_level, burn, num_avg, max_lag_UB, skip_first, horizon,
                                  include_X, bool_use_anomalies, bool_use_seas_dummies, bool_use_PCA, ar_lags, num_comps, setting, method, bool_use_pre_estim_ast,bool_use_pre_estim_noexog)
            MSE_mat[i,h] = mse
        config_names.append(config_name)

start = timeit.default_timer()
run_forecasting_configs(configurations)
end = timeit.default_timer()
print(f"Total computation time: {end - start:.2f}s")

df = pd.DataFrame(
    MSE_mat,
    index=config_names,
    columns=horizons
)
print(df)

