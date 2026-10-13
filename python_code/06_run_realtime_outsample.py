#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Janneke van Brummelen
"""
from python_code.methods.Outsample_estimation import *
from dateutil.relativedelta import relativedelta

def StringToDateExtend(string_dates, horizons):
    """
    Purpose:
        Convert string to datetime.date object and extend by horizon

    Inputs:
        vIntDate    vector, dates as strings (example: 2016-06-18)

    Return value:
        vNumDates   vector, dates converted to days since 01-01-01
        vsStringDates   vector, reformatted date strings in ymd format
    """
    N = np.size(string_dates)
    num_dates = np.zeros((N, 1))
    list_string_dates = []

    for i in range(N):
        date = string_dates[i]
        # Current configuration is day-month-year
        year = int(date[:4])
        month = int(date[5:])
        date_time = dt.date(year, month, 1)
        num_dates[i] = mdates.date2num(date_time)
        list_string_dates.append(date[:4] + "-" + date[5:] + "-01")

    forecast_dates = []
    for h in horizons:
        date_time_forecast_date = date_time + relativedelta(months=+h)
        forecast_dates += [str(date_time_forecast_date.year) + "-" + str(date_time_forecast_date.month) + "-" + str(
            date_time_forecast_date.day)]

    return num_dates, np.array(list_string_dates), np.array(forecast_dates)


####################
##### Settings #####
####################
output_folder = "../output/intermediary_data/realoutsample/"
table_output_folder = "../output/tables/"
table_name = "TableA12_realtime_forecasts"

input_folder_name = "../data"
data_file_name = "Book1982_2025_autosim_HIGHRES_OISST"
date_name = "DATE"

bool_use_dates = True
bool_use_random_search = True
bool_use_anomalies = False
horizons = [1, 3, 6, 12, 14, 16, 18]

bool_use_pre_estim = True
bool_save_params = True

burn = 6
num_avg = 6
sig_level = 0.01

skip_last = 2
skip_first = 238
fin_month_index = 9
months_prior_plot = 10 # number of EN anomalies to include in plot of the real-time forecasts, should be >= 1

####################
### Data loading ###
####################
Y_label = 'EN3.4'
X_labels = ['RB', 'WPAC', 'WPAC2', 'WPAC3', 'WPAC4', '50fin',
            '100cold', '100fin1', '100fin2', '150fin1', '150fin2', '200fin1',
            '200fin2', '250fin1', '250fin2', '300fin1', '300fin2', '400fin1',
            '400fin2', '500fin1', '500fin2', 'wnd160.200_0.10', 'wnd180.220_-4.4', 'wnd180.210_-10.0']

insample_length = 276

# build lag lists
max_lag_UB = 36
X_min_lags = [6, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6]
X_max_lags = [11, 5, 5, 5, 5, 5, 11, 11, 11, 11, 18, 11, 5, 18, 18, 11, 18, 5, 18, 11, 18, 11, 11, 18]
X_max_lags = [max(x_max, x_min - 1) for x_max, x_min in zip(X_max_lags, X_min_lags)]

X_lags = []
for i, lab in enumerate(X_labels):
    lags = [j for j in range(X_min_lags[i], X_max_lags[i] + 1)]
    if lab in ['WPAC', 'WPAC2', 'WPAC3', 'WPAC4', '50fin1', '400fin1', '200fin2']:
        lags = lags + [j for j in range(12, 19)]
    if lab in ['50fin', '100fin1', '150fin1', '200fin1', '250fin1', '300fin1', '400fin11', '500fin1', '100fin2',
               '150fin2', '200fin2', '250 fin2', '300fin2', '100cold', 'wnd160.200_0.10', 'wnd180.220_-4.4',
               'wnd180.210_-10.0']:
        lags = lags + [j for j in range(19, 37)]
    X_lags = X_lags + [np.array(lags)]

################### Reading and transforming data #######################
dfData = pd.read_excel(input_folder_name + "/" + data_file_name + ".xlsx", na_values='.', header=0)
lTotLabels = [Y_label] + X_labels

data_matrix = dfData[lTotLabels].values
Y_mat = data_matrix.astype(float)
N = Y_mat.shape[1]
string_dates = dfData[date_name].values

T_tot = Y_mat.shape[0]

monthly_means = np.array([np.mean(Y_mat[i::12,0]) for i in range(12)])
repeated_means = np.tile(monthly_means, int(np.ceil(T_tot / 12)))
Y_vec_anom = Y_mat[:,0].reshape(-1, 1) - repeated_means[:T_tot].reshape(-1, 1)
demean_vector = [monthly_means[(h+fin_month_index)%12] for h in horizons]

if skip_first>0:
    Y_mat = Y_mat[skip_first:, :]
    string_dates = string_dates[skip_first:]
if skip_last > 0:
    Y_mat = Y_mat[:-skip_last, :]
    string_dates = string_dates[:-skip_last]
    Y_vec_anom = Y_vec_anom[:-skip_last]
if bool_use_dates:
    num_dates, string_dates, forecast_dates = StringToDateExtend(string_dates, horizons)

Y_vec = Y_mat[:, 0]
X_vars = Y_mat[:, 1:]
if insample_length != Y_vec.shape[0]:
    print("Not the right sample length")
print(string_dates[0],string_dates[-1])

last_Ys_anom = Y_vec_anom[-months_prior_plot:]
#########################################################################
######################### CONFIGURATIONS ################################
#########################################################################
configurations = []

for num_comps, setting in [
    (0, 'A'),
    (3, 'B'),
    (3, 'C'),
    (5, 'B'),
    (5, 'C'),
]:
    configurations.append({
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
# constant models
for method in ["G2S", "Lasso"]:
    for bool_use_PCA in [False, True]:
        configurations.append({
            "include_X": True,
            "method": method,
            "num_comps": 0,
            "setting": "A",
            "ar_lags": 0,
            "bool_use_seas_dummies": False,
            "bool_use_PCA": bool_use_PCA
        })

# seasonal + cycle models
for bool_use_PCA in [False, True]:
    configurations.append({
        "include_X": True,
        "method": "G2S",
        "num_comps": 5,
        "setting": "B",
        "ar_lags": 0,
        "bool_use_seas_dummies": False,
        "bool_use_PCA": bool_use_PCA
    })

#########################################################################
#########################################################################
########################### ESTIMATION ##################################
#########################################################################
#########################################################################
def run_configs(configurations):
    forecasts = np.zeros((len(horizons),len(configurations)))
    string_info_vec = []

    for i, config in enumerate(configurations):
        include_X = config["include_X"]
        method = config["method"]
        num_comps = config["num_comps"]
        setting = config["setting"]
        ar_lags = config["ar_lags"]
        bool_use_seas_dummies = config["bool_use_seas_dummies"]
        bool_use_PCA = config["bool_use_PCA"]

        for j, horizon in enumerate(horizons):
            Y_vec_ast = np.vstack((Y_vec.reshape(-1, 1), np.zeros((horizon, 1)))).reshape(-1, )
            if include_X:
                X_vars_ast = X_vars.copy()
                X_vars_ast = np.vstack((X_vars_ast,np.zeros((horizon,X_vars.shape[1]))))
            else:
                X_vars_ast = np.zeros((Y_vec_ast.shape[0], 0))
            if horizon > 1 and not include_X:
                bool_use_pre_estim_ast = True
            else:
                bool_use_pre_estim_ast = bool_use_pre_estim
            if include_X and num_comps > 0:
                bool_use_pre_estim_noexog = True
            else:
                bool_use_pre_estim_noexog = False
            _, string_info, forec = EstimationAndForecast(output_folder, insample_length, Y_vec_ast, X_vars_ast, num_dates, string_dates,
                                  X_lags, X_labels, bool_use_random_search, bool_use_dates, sig_level, burn, num_avg, max_lag_UB,
                                  skip_first, horizon, include_X, bool_use_anomalies, bool_use_seas_dummies, bool_use_PCA, ar_lags,
                                  num_comps, setting, method, bool_use_pre_estim_ast, bool_use_pre_estim_noexog,save_results=bool_save_params)

            forecasts[j, i] = forec[0]
            print('horizon',horizon,':', forecasts[j,i])
        string_info_vec += [string_info]
    return forecasts, string_info_vec

# run configurations: saved automatically
start = timeit.default_timer()
forecasts, string_info_vec = run_configs(configurations)
stop = timeit.default_timer()
print(f"Estimation time models without regressors: {stop - start:.2f}s")

### Save forecasts and forecast errors for plotting
df_forecasts =  pd.DataFrame(forecasts,columns=string_info_vec,index=forecast_dates)
with open(output_folder+'forecasts.csv','w') as tf:
    tf.write(df_forecasts.to_csv())

demean_forecasts = forecasts - np.array(demean_vector).reshape(len(horizons),1)
demean_forecasts_ext = np.vstack((last_Ys_anom[-1] *np.ones((1,demean_forecasts.shape[1])), demean_forecasts))
index = np.concatenate((string_dates[-months_prior_plot:],forecast_dates))
df_demean_forecasts =  pd.DataFrame(index=index,columns=["EN"] + string_info_vec)
df_demean_forecasts.loc[np.concatenate(([string_dates[-1]],forecast_dates)), string_info_vec] = demean_forecasts_ext
df_demean_forecasts.loc[string_dates[-months_prior_plot:], "EN"] = last_Ys_anom.ravel()

print(df_demean_forecasts)
with open(output_folder+'demean_forecasts.csv','w') as tf:
    tf.write(df_demean_forecasts.to_csv())

### Save results in a LaTeX table
row_labels = [
    ("Levels", None),
    ("Only constant", 0),

    ("Regressors", None),
    ("\\quad \\textit{G2S}", 5),
    ("\\quad \\textit{G2S PCA}", 6),
    ("\\quad \\textit{LASSO}", 7),
    ("\\quad \\textit{LASSO PCA}", 8),

    ("Seasonals", None),
    ("\\quad \\textit{constant}", 1),
    ("\\quad \\textit{trend}", 2),

    ("Seasonals + cycles", None),
    ("\\quad \\textit{constant}", 3),
    ("\\quad \\textit{trend}", 4),

    ("Seasonals + cycles + regressors", None),
    ("\\quad \\textit{G2S}", 9),
    ("\\quad \\textit{G2S PCA}", 10),
]

def write_panel(f, title, data):
    f.write("\\midrule\n")
    f.write(f"\\textbf{{{title}}}\\\\\n")

    for label, idx in row_labels:
        if idx is None:
            f.write("\\\\\n")
            f.write(label + " \\\\\n")
        else:
            vals = " & ".join(f"{x:.4f}" for x in data[:, idx])
            f.write(f"{label} & {vals}\\\\\n")

with open(table_output_folder  + table_name + ".tex", "w") as f:

    f.write("\\begin{tabular}{l" + "c"*len(horizons) + "}\n")
    f.write("\\toprule\n")

    f.write("& " +
            " & ".join(f"$h={h}$" for h in horizons)
            + "\\\\\n")

    months = [d[5:7] + "-'" + d[2:4] for d in forecast_dates]
    f.write("& " + " & ".join(months) + "\\\\\n")

    write_panel(f, "Levels", forecasts)
    write_panel(f, "Anomalies", demean_forecasts)

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")