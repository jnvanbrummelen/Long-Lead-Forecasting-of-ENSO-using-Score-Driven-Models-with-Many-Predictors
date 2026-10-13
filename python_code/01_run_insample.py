#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
""" 
@author: Janneke van Brummelen
"""
from python_code.methods.Insample_estimation import *

####################
##### Settings #####
####################
output_folder = "../output/intermediary_data/insample/"
input_folder_name = "../data"
data_file_name = "Book1982_2025_autosim_HIGHRES_OISST"

bool_use_dates = True
bool_use_random_search = True
bool_use_anomalies = False
horizons = [1, 6, 12, 18]

burn = 6
num_avg = 6
sig_level = 0.01

####################
### Data loading ###
####################
date_name = "DATE"
Y_label = 'EN3.4'
X_labels = ['RB', 'WPAC', 'WPAC2', 'WPAC3', 'WPAC4', '50fin',
       '100cold', '100fin1', '100fin2', '150fin1', '150fin2', '200fin1',
       '200fin2', '250fin1', '250fin2', '300fin1', '300fin2', '400fin1',
       '400fin2', '500fin1', '500fin2', 'wnd160.200_0.10', 'wnd180.220_-4.4', 'wnd180.210_-10.0']

insample_length = 276
skip_last = 0
vec_skip_first = [0, 240]

# build lag lists
max_lag_UB = 36   
X_min_lags = [6,  1,  1,  1,  1,  1,  6,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1, 1,  1,  1,  1,  6,  6,  6]
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

if skip_last > 0:
    Y_mat = Y_mat[:-skip_last,:] 
    string_dates = string_dates[:-skip_last]
if bool_use_dates:
    num_dates, string_dates = StringToDate(string_dates)
 
Y_vec = Y_mat[:,0]
X_vars = Y_mat[:,1:]

monthly_means = np.array([np.mean(Y_vec[i::12]) for i in range(12)])
df_monthly_means = pd.DataFrame(monthly_means,index=np.arange(1,13),columns=['monthly mean'])
with open(output_folder+'monthly_means.csv','w') as tf:
    tf.write(df_monthly_means.to_csv())

#########################################################################
######################### CONFIGURATIONS ################################
#########################################################################
configurations_noexog = []

for skip_first in vec_skip_first:
    for num_comps, setting in [
        (0, 'A'),
        (3, 'B'),
        (3, 'C'),
        (5, 'B'),
        (5, 'C'),
    ]:

        configurations_noexog.append({
            "horizon": 1,
            "include_X": False,
            "method": "",
            "num_comps": num_comps,
            "ar_lags": 0,
            "setting": setting,
            "bool_use_seas_dummies": False,
            "bool_use_PCA": False,
            "skip_first": skip_first,
        })

## configurations models with exogenous regressors
horizons = [1, 6, 12, 18]

configurations_exog = []

# constant models
for horizon in horizons:
    for method in ["G2S", "Lasso"]:
        for bool_use_PCA in [False, True]:
            configurations_exog.append({
                "horizon": horizon,
                "include_X": True,
                "method": method,
                "num_comps": 0,
                "setting": "A",
                "ar_lags": 0,
                "bool_use_seas_dummies": False,
                "bool_use_PCA": bool_use_PCA,
                "skip_first": vec_skip_first[0]
            })

# seasonal + cycle models
for horizon in horizons:
    for bool_use_PCA in [False, True]:
        configurations_exog.append({
            "horizon": horizon,
            "include_X": True,
            "method": "G2S",
            "num_comps": 5,
            "setting": "B",
            "ar_lags": 0,
            "bool_use_seas_dummies": False,
            "bool_use_PCA": bool_use_PCA,
            "skip_first": vec_skip_first[0]
        })

#########################################################################
#########################################################################
########################### ESTIMATION ##################################
#########################################################################
#########################################################################
def run_configs(configurations):
    for config in configurations:
        horizon = config["horizon"]
        include_X = config["include_X"]
        method = config["method"]
        num_comps = config["num_comps"]
        setting = config["setting"]
        ar_lags = config["ar_lags"]
        bool_use_seas_dummies = config["bool_use_seas_dummies"]
        bool_use_PCA = config["bool_use_PCA"]
        skip_first = config["skip_first"]

        if skip_first > 0:
            Y_vec_ast = Y_vec[skip_first:]
            X_vars_ast = X_vars[skip_first:,:]
            string_dates_ast = string_dates[skip_first:]
            num_dates_ast = num_dates[skip_first:]
        else:
            Y_vec_ast = Y_vec
            X_vars_ast = X_vars
            string_dates_ast = string_dates
            num_dates_ast = num_dates
        if insample_length > 0:
            Y_vec_ast = Y_vec_ast[:insample_length]
            X_vars_ast = X_vars_ast[:insample_length,:]
            string_dates_ast = string_dates_ast[:insample_length]
            num_dates_ast = num_dates_ast[:insample_length]

        if ar_lags > 0:
            lYlags = [i for i in range(1,ar_lags+1)]
            X_lags_ast = [np.array(lYlags)] + X_lags
        else:
            X_lags_ast = X_lags

        if bool_use_anomalies:
            monthly_means = np.array([np.mean(Y_vec_ast[i::12]) for i in range(12)])
            repeated_means = np.tile(monthly_means,int(np.ceil(T_tot/12)))
            Y_vec_ast = Y_vec_ast.reshape(-1,1) - repeated_means[:Y_vec_ast.shape[0]].reshape(-1,1)

        Parameter_estimation(output_folder, Y_vec_ast, X_vars_ast, num_dates_ast, string_dates_ast, X_lags_ast, X_labels,
                             bool_use_random_search, bool_use_dates, sig_level, burn, num_avg,
                             max_lag_UB, skip_first, horizon, include_X, bool_use_anomalies,
                             bool_use_seas_dummies, bool_use_PCA, ar_lags, num_comps, setting, method)

# run configurations: saved automatically
start = timeit.default_timer()
run_configs(configurations_noexog)
stop = timeit.default_timer()
print(f"Estimation time models without regressors: {stop - start:.2f}s")

start = timeit.default_timer()
run_configs(configurations_exog)
stop = timeit.default_timer()
print(f"Estimation time models with regressors: {stop - start:.2f}s")