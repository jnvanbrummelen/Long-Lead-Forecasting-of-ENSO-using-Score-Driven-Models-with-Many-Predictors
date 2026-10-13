import os
import pandas as pd
import numpy as np

##############################################################################
# SETTINGS
##############################################################################
forecast_error_folder = "../output/intermediary_data/outsample/forecast_errors"
output_folder = "../output/intermediary_data/outsample/combined_forecast_errors"
os.makedirs(output_folder, exist_ok=True)

# horizons to combine
horizons = [1, 3, 6, 12, 14, 16, 18]

##############################################################################
# CONFIGURATION GENERATOR
##############################################################################

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
# With only exogenous variables
##########################################################################

for method in ['G2S', 'Lasso']:
    for bool_use_PCA in [False, True]:
        configurations.append({
            "include_X": True,
            "method": method,
            "num_comps": 0,
            "ar_lags": 0,
            "setting": 'A',
            "bool_use_seas_dummies": False,
            "bool_use_PCA": bool_use_PCA,
        })

##########################################################################
# Components and exogenous variables
##########################################################################

for num_comps, bool_use_PCA in [
    (3, False),
    (3, True),
    (5, False),
    (5, True),
]:

    for bool_use_seas_dummies in [False, True]:
        if num_comps == 3 and not bool_use_seas_dummies:
            continue
        configurations.append({
            "include_X": True,
            "method": 'G2S',
            "num_comps": num_comps,
            "ar_lags": 0,
            "setting": 'B',
            "bool_use_seas_dummies": bool_use_seas_dummies,
            "bool_use_PCA": bool_use_PCA,
        })

##########################################################################
# AR(3) models
##########################################################################
configurations.append({
            "include_X": False,
            "method": '',
            "num_comps": 0,
            "ar_lags": 3,
            "setting": 'A',
            "bool_use_seas_dummies": False,
            "bool_use_PCA": False,
        })

for bool_use_PCA in [
    False,
    True
]:

    configurations.append({
            "include_X": True,
            "method": 'G2S',
            "num_comps": 0,
            "ar_lags": 3,
            "setting": 'A',
            "bool_use_seas_dummies": False,
            "bool_use_PCA": bool_use_PCA,
        })

##########################################################################
# AR(1) models
##########################################################################
configurations.append({
            "include_X": False,
            "method": '',
            "num_comps": 0,
            "ar_lags": 1,
            "setting": 'A',
            "bool_use_seas_dummies": False,
            "bool_use_PCA": False,
        })

for bool_use_PCA in [
    False,
    True
]:

    configurations.append({
            "include_X": True,
            "method": 'G2S',
            "num_comps": 0,
            "ar_lags": 1,
            "setting": 'A',
            "bool_use_seas_dummies": False,
            "bool_use_PCA": bool_use_PCA,
        })

##############################################################################
# STRING BUILDER
##############################################################################

def build_string_info(config, horizon):

    include_X = config["include_X"]
    method = config["method"]
    num_comps = config["num_comps"]
    ar_lags = config["ar_lags"]
    setting = config["setting"]
    bool_use_seas_dummies = config["bool_use_seas_dummies"]
    bool_use_PCA = config["bool_use_PCA"]

    str_ar_lags = '' if (ar_lags == 3 or ar_lags == 0) else str(ar_lags)
    bool_use_AR_lags = ar_lags > 0

    string_info = (
        method * include_X
        + str(num_comps)
        + 'componentsSetting'
        + setting
        + bool_use_AR_lags * ('_AR' + str_ar_lags)
        + bool_use_seas_dummies * '_seasdummies'
        + bool_use_AR_lags * '_anomalies'
        + 'withExog' * include_X
        + bool_use_PCA * 'PCA'
        + 'horizon'
        + str(horizon)
    )

    return string_info

##############################################################################
# MAIN LOOP
##############################################################################
msfe_results = {}
for horizon in horizons:

    print(f"\n===================================================")
    print(f"Processing horizon {horizon}")
    print(f"===================================================")

    merged_df = None

    for config in configurations:

        string_info = build_string_info(config, horizon)

        file_name = f"forecast_errors{string_info}.csv"

        file_path = os.path.join(
            forecast_error_folder,
            file_name
        )

        if not os.path.exists(file_path):
            print(f"Missing file: {file_name}")
            continue

        print(f"Reading: {file_name}")

        df = pd.read_csv(file_path)

        dates = pd.to_datetime(df.iloc[:, 0])
        errors = df.iloc[:, 1]

        msfe = (errors ** 2).mean()
        model_label = string_info.replace(f'horizon{horizon}', '')
        if model_label not in msfe_results:
            msfe_results[model_label] = {}
        msfe_results[model_label][horizon] = msfe

        temp_df = pd.DataFrame({
            "Date": dates,
            string_info: errors
        })

        if merged_df is None:

            merged_df = temp_df

        else:

            merged_df = pd.merge(
                merged_df,
                temp_df,
                on="Date",
                how="outer"
            )

    ##########################################################################
    # SAVE
    ##########################################################################

    if merged_df is not None:

        merged_df.sort_values("Date", inplace=True)

        output_file = os.path.join(
            output_folder,
            f"combined_forecast_errors_horizon{horizon}.csv"
        )

        merged_df.to_csv(output_file, index=False)

        print(f"\nSaved combined file:")
        print(output_file)

    else:

        print(f"No files found for horizon {horizon}")

##############################################################################
# CREATE MSFE TABLE
##############################################################################

msfe_df = pd.DataFrame(msfe_results).T

# ensure horizon ordering
msfe_df = msfe_df[horizons]

msfe_df.columns = [f"h={h}" for h in horizons]

print("\n===================================================")
print("MEAN SQUARED FORECAST ERRORS (MSFE)")
print("===================================================")

print(msfe_df.round(4))

##############################################################################
# SAVE MSFE TABLE
##############################################################################

msfe_output_file = os.path.join(
    output_folder,
    "MSFE_results.csv"
)

msfe_df.to_csv(msfe_output_file)

print(f"\nSaved MSFE table:")
print(msfe_output_file)

print("\nDone.")

##############################################################################
##############################################################################
# SAVING THE ANOMALY FORECASTS FOR FIGURE 3
##############################################################################
##############################################################################

##############################################################################
# SETTINGS
##############################################################################
horizons_anomaly = [6, 12, 18]
forecast_folder = "../output/intermediary_data/outsample/forecasts"
output_folder_forecasts = "../output/intermediary_data/outsample/anomaly_forecasts"
os.makedirs(output_folder_forecasts, exist_ok=True)

configurations_anomaly = [
    {"include_X": True, "method": 'Lasso', "num_comps": 0, "ar_lags": 0,
     "setting": 'A', "bool_use_seas_dummies": False, "bool_use_PCA": False},
    {"include_X": True, "method": 'G2S', "num_comps": 0, "ar_lags": 0,
     "setting": 'A', "bool_use_seas_dummies": False, "bool_use_PCA": True},
    {"include_X": False, "method": "", "num_comps": 5, "ar_lags": 0,
     "setting": 'B', "bool_use_seas_dummies": False, "bool_use_PCA": False},
    {"include_X": True, "method": 'G2S', "num_comps": 5, "ar_lags": 0,
     "setting": 'B', "bool_use_seas_dummies": False, "bool_use_PCA": True},
]

##############################################################################
# LOAD RAW DATA AND COMPUTE MONTHLY MEANS
##############################################################################
folder_name =  "../data"
data_file_name = "Book1982_2025_autosim_HIGHRES_OISST"
date_name = "DATE"
Y_label = 'EN3.4'

fData = pd.read_excel(folder_name + "/" + data_file_name + ".xlsx",
                      na_values='.', header=0)

# parse dates
string_dates_raw = fData[date_name].values
Y_raw = fData[Y_label].values.astype(float)

# compute monthly means over full sample (month 0=Jan, ..., 11=Dec)
monthly_means = np.array([np.mean(Y_raw[i::12]) for i in range(12)])

# build a date index aligned with Y_raw for month lookup
dates_pd = pd.to_datetime(string_dates_raw)
date_to_month = {d: d.month - 1 for d in dates_pd}  # 0-indexed month

##############################################################################
# COMPUTE ANOMALY OF DEPENDENT VARIABLE
##############################################################################
T_tot = len(Y_raw)
repeated_means = np.tile(monthly_means, int(np.ceil(T_tot / 12)))[:T_tot]
Y_anomaly = Y_raw - repeated_means

df_Y_anomaly = pd.DataFrame({
    "Date": dates_pd,
    "Y_anomaly": Y_anomaly
})

##############################################################################
# MAIN LOOP
##############################################################################

for horizon in horizons_anomaly:

    print(f"\n=== Horizon {horizon} ===")

    merged_df = None

    for config in configurations_anomaly:

        string_info = build_string_info(config, horizon)

        file_path = os.path.join(
            forecast_folder,
            f"forecasts{string_info}.csv"
        )

        if not os.path.exists(file_path):
            print(f"Missing: forecasts{string_info}.csv")
            continue

        print(f"Reading: forecasts{string_info}.csv")

        df = pd.read_csv(file_path)

        # keep full dates
        forecast_dates = pd.to_datetime(df.iloc[:, 0])

        forecasts = df.iloc[:, 1].values

        # month indices: Jan=0, ..., Dec=11
        forecast_months = forecast_dates.dt.month.values - 1

        forecasts_anomaly = forecasts - monthly_means[forecast_months]

        model_label = string_info.replace(f'horizon{horizon}', '')

        temp_df = pd.DataFrame({
            "Date": forecast_dates,
            model_label: forecasts_anomaly
        })

        if merged_df is None:

            merged_df = temp_df

        else:

            merged_df = pd.merge(
                merged_df,
                temp_df,
                on="Date",
                how="outer"
            )

    # attach Y anomaly only for retained forecast dates
    merged_df = pd.merge(
        merged_df,
        df_Y_anomaly,
        on="Date",
        how="left"
    )

    merged_df.rename(
        columns={"Y_anomaly": "EN3.4"},
        inplace=True
    )

    # put EN3.4 first
    other_cols = [c for c in merged_df.columns if c not in ["Date", "EN3.4"]]

    merged_df = merged_df[
        ["Date", "EN3.4"] + other_cols
        ]

    merged_df.sort_values("Date", inplace=True)

    ######################################################################
    # SAVE
    ######################################################################

    output_file = os.path.join(
        output_folder_forecasts,
        f"anomaly_forecasts_horizon{horizon}.csv"
    )

    merged_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")













