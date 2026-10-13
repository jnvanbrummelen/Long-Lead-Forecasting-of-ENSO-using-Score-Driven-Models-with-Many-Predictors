import os
import pandas as pd

##############################################################################4
# SETTINGS
##############################################################################
parameter_folder = "../output/intermediary_data/insample/parameters"
output_folder = "../output/tables"
os.makedirs(output_folder, exist_ok=True)

table_names = ["Table1_insample_without_regressors","TableA4_insample_without_regressors_subsample2",
               "Table2_insample_with_regressors","TableA5_insample_selected_vars_with_reg_h1_h6",
               "TableA6_insample_selected_vars_with_reg_h12_h18","TableA7_insample_selected_vars_with_reg_PCA"]
vec_skip_first = [0, 240]

##############################################################################
# CONFIGURATIONS
##############################################################################

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
    })


##############################################################################
# STRING BUILDER
##############################################################################

def build_string_info(config):

    include_X = config["include_X"]
    method = config["method"]

    num_comps = config["num_comps"]
    setting = config["setting"]

    ar_lags = config["ar_lags"]

    bool_use_seas_dummies = config["bool_use_seas_dummies"]
    bool_use_PCA = config["bool_use_PCA"]

    bool_use_AR_lags = ar_lags > 0
    bool_use_anomalies = bool_use_AR_lags

    ##########################################################################
    # BASE STRING
    ##########################################################################

    string_info = (
        str(num_comps)
        + "componentsSetting"
        + setting
        + bool_use_AR_lags * "_AR"
        + bool_use_seas_dummies * "_seasdummies"
        + bool_use_anomalies * "_anomalies"
    )

    ##########################################################################
    # EXOGENOUS MODELS
    ##########################################################################

    if include_X:

        horizon = config["horizon"]

        string_info += "_horizon" + str(horizon)

        string_info += "withExog"

        if bool_use_PCA:
            string_info += "PCA"

        string_info += "_" + method

    ##########################################################################
    # SKIP FIRST
    ##########################################################################

    skip_first = config.get("skip_first", 0)

    if skip_first > 0:
        string_info += "skipfirst" + str(skip_first)

    return string_info

##############################################################################
# LOAD ALL MODELS
##############################################################################
for skip_first in vec_skip_first:
    models = []
    data = {}

    for config in configurations:
        config["skip_first"] = skip_first
        label = build_string_info(config)

        file_path = os.path.join(parameter_folder, f"params{label}.csv")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        print(f"Reading: {file_path}")

        df = pd.read_csv(file_path)

        models.append(label)

        ##########################################################################
        # STORE PARAMS
        ##########################################################################

        data[label] = {}

        for _, row in df.iterrows():

            param_name = str(row["Param"]).strip()

            ######################################################################
            # detect fixed parameters
            ######################################################################

            is_fixed = "(fixed)" in param_name

            ######################################################################
            # clean parameter name
            ######################################################################

            clean_name = (
                param_name
                .replace(" (fixed)", "")
                .strip()
            )

            ######################################################################
            # standard errors
            ######################################################################

            se = row["Std. error"]

            if is_fixed:
                se = "n/a"

            data[label][clean_name] = (
                row["Est"],
                se
            )

    ##############################################################################
    # PARAMETER ORDER
    ##############################################################################

    params = [

        "sigma^2",

        "omega",
        "theta_1_1",

        "phi_2_1",
        "phi_2_2",
        "theta_2_1",

        "phi_3_1",
        "phi_3_2",
        "theta_3_1",

        "phi_4_1",
        "phi_4_2",
        "theta_4_1",

        "phi_5_1",
        "phi_5_2",
        "theta_5_1",
    ]

    ##############################################################################
    # LATEX NAMES
    ##############################################################################

    latex_map = {

        ##########################################################################
        # variance + level
        ##########################################################################

        "sigma^2": r"$\sigma^2$",

        "omega": r"$\alpha$",

        "theta_1_1": r"$\theta_{\alpha}$",

        ##########################################################################
        # seasonal block 1
        ##########################################################################

        "phi_2_1": r"$\phi_{\gamma,1}^{(1)}$",

        "phi_2_2": r"$\phi_{\gamma,2}^{(1)}$",

        "theta_2_1": r"$\theta_{\gamma}^{(1)}$",

        ##########################################################################
        # seasonal block 2
        ##########################################################################

        "phi_3_1": r"$\phi_{\gamma,1}^{(2)}$",

        "phi_3_2": r"$\phi_{\gamma,2}^{(2)}$",

        "theta_3_1": r"$\theta_{\gamma}^{(2)}$",

        ##########################################################################
        # cycle block 1
        ##########################################################################

        "phi_4_1": r"$\phi_{\psi,1}^{(1)}$",

        "phi_4_2": r"$\phi_{\psi,2}^{(1)}$",

        "theta_4_1": r"$\theta_{\psi}^{(1)}$",

        ##########################################################################
        # cycle block 2
        ##########################################################################

        "phi_5_1": r"$\phi_{\psi,1}^{(2)}$",

        "phi_5_2": r"$\phi_{\psi,2}^{(2)}$",

        "theta_5_1": r"$\theta_{\psi}^{(2)}$",
    }

    ##############################################################################
    # BUILD LATEX TABLE
    ##############################################################################

    latex_lines = []

    ##############################################################################
    # PARAMETER PANEL
    ##############################################################################

    for p in params:

        est_row = [latex_map.get(p, p)]
        se_row = [""]

        for m in models:

            if p in data[m]:

                est, se = data[m][p]

                ##################################################################
                # estimate
                ##################################################################

                if pd.isna(est):
                    est_row.append("")
                else:
                    est_row.append(f"{est:.3f}")

                ##################################################################
                # standard error
                ##################################################################

                if isinstance(se, str):

                    se_row.append(
                        r"{\footnotesize\textit{" + se + "}}"
                    )

                elif pd.isna(se):

                    se_row.append("")

                else:

                    se_row.append(
                        r"{\footnotesize\textit{" + f"{se:.3f}" + "}}"
                    )

            else:

                est_row.append("")
                se_row.append("")

        latex_lines.append(" & ".join(est_row) + r" \\")
        latex_lines.append(" & ".join(se_row) + r" \\")

    ##############################################################################
    # PERIODICITIES PANEL
    ##############################################################################

    latex_lines.append(r"\midrule")

    period_rows = [
        "period_2",
        "period_3",
        "period_4",
        "period_5",
    ]

    period_map = {

        "period_2": r"Periodicity $\gamma_t^{(1)}$",

        "period_3": r"Periodicity $\gamma_t^{(2)}$",

        "period_4": r"Periodicity $\psi_t^{(1)}$",

        "period_5": r"Periodicity $\psi_t^{(2)}$",
    }

    for p in period_rows:

        row = [period_map[p]]

        for m in models:

            if p in data[m]:

                val = data[m][p][0]

                if pd.isna(val):
                    row.append("")
                else:
                    row.append(f"{val:.3f}")

            else:
                row.append("")

        latex_lines.append(" & ".join(row) + r" \\")

    ##############################################################################
    # FIT STATISTICS PANEL
    ##############################################################################

    latex_lines.append(r"\midrule")

    stats = ["LLH", "AIC", "BIC"]

    for s in stats:

        row = [s]

        for m in models:

            if s in data[m]:

                val = data[m][s][0]

                if pd.isna(val):
                    row.append("")
                else:
                    row.append(f"{val:.2f}")

            else:
                row.append("")

        latex_lines.append(" & ".join(row) + r" \\")


    ##############################################################################
    # filename
    ##############################################################################
    if skip_first > 0:
        file_name = table_names[1]
    else:
        file_name = table_names[0]

    latex_file = os.path.join(
        output_folder,
        f"{file_name}.tex"
    )

    ##############################################################################
    # WRITE LATEX
    ##############################################################################

    with open(latex_file, "w") as f:

        f.write(r"\begin{table}[t!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\footnotesize" + "\n")

        f.write(
            r"\begin{tabular}{l"
            + "c" * len(models)
            + "}"
            + "\n"
        )

        f.write(r"\toprule" + "\n")

        ##########################################################################
        # COLUMN HEADERS
        ##########################################################################

        f.write(
            r" & constant & constant & trend & constant & trend \\"
            + "\n"
        )

        f.write(r"\midrule" + "\n")

        ##########################################################################
        # TABLE BODY
        ##########################################################################

        for line in latex_lines:
            f.write(line + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print("\nSaved LaTeX table to:")
    print(latex_file)



##############################################################################
##############################################################################
# AGGREGATED EXOGENOUS MODEL TABLE
##############################################################################
##############################################################################


##############################################################################
# CONFIGURATIONS
##############################################################################

horizons = [1, 6, 12, 18]

configurations_exog = []

##########################################################################
# constant models
##########################################################################

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

##########################################################################
# seasonal + cycle models
##########################################################################

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

##############################################################################
# COLUMN ORDER
##############################################################################

column_order = [

    ("G2S", False, 0),
    ("Lasso", False, 0),

    ("G2S", True, 0),
    ("Lasso", True, 0),

    ("G2S", False, 5),
    ("G2S", True, 5),
]

##############################################################################
# LOAD RESULTS
##############################################################################

summary_stats = {}

for config in configurations_exog:

    label = build_string_info(config)

    file_path = os.path.join(
        parameter_folder,
        f"params{label}.csv"
    )

    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        continue

    print(f"Reading: {file_path}")

    df = pd.read_csv(file_path)

    horizon = config["horizon"]

    key = (
        config["method"],
        config["bool_use_PCA"],
        config["num_comps"]
    )

    if horizon not in summary_stats:
        summary_stats[horizon] = {}

    summary_stats[horizon][key] = {}

    ##########################################################################
    # extract stats
    ##########################################################################

    for _, row in df.iterrows():

        param = str(row["Param"]).strip()



        if param in [
            "LLH",
            "AIC",
            "BIC",
            "no. selected vars",
            "total vars"
        ]:

            summary_stats[horizon][key][param] = row["Est"]

##############################################################################
# BUILD LATEX
##############################################################################

latex_lines_exog = []

stats_order = [
    "LLH",
    "AIC",
    "BIC",
    "no. selected vars",
    "total vars",
]

for horizon in horizons:

    latex_lines_exog.append(
        r"\textit{horizon "
        + str(horizon)
        + r"} & & & & & & \\"
    )

    for stat in stats_order:

        row = [stat]

        for key in column_order:

            if (
                horizon in summary_stats
                and key in summary_stats[horizon]
                and stat in summary_stats[horizon][key]
            ):

                val = summary_stats[horizon][key][stat]

                if stat in ["LLH", "AIC", "BIC"]:
                    row.append(f"{val:.2f}")
                else:
                    row.append(f"{int(val)}")

            else:
                row.append("")

        latex_lines_exog.append(
            " & ".join(row) + r" \\"
        )

    latex_lines_exog.append(
        r" & & & & & & \\"
    )

##############################################################################
# SAVE LATEX
##############################################################################


latex_file_exog = os.path.join(
    output_folder,
    f"{table_names[2]}.tex"
)

with open(latex_file_exog, "w") as f:

    f.write(r"\begin{table}[t!]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\small" + "\n")

    f.write(
        r"\begin{tabularx}{1.00\textwidth}"
        r"{@{\extracolsep{\stretch{1}}}lrrrrrr@{}}"
        + "\n"
    )

    f.write(r"\toprule" + "\n")

    ##########################################################################
    # HEADER
    ##########################################################################

    f.write(
        r"& \multicolumn{4}{c}{constant}"
        r"& \multicolumn{2}{c}{seasonals + cycles} \\"
        + "\n"
    )

    f.write(
        r"\cmidrule(lr){2-5}"
        r"\cmidrule(lr){6-7}"
        + "\n"
    )

    f.write(
        r"& \multicolumn{2}{c}{no PCA}"
        r"& \multicolumn{2}{c}{PCA}"
        r"& \multicolumn{1}{c}{no PCA}"
        r"& \multicolumn{1}{c}{PCA} \\"
        + "\n"
    )

    f.write(
        r"\cmidrule(lr){2-3}"
        r"\cmidrule(lr){4-5}"
        r"\cmidrule(lr){6-6}"
        r"\cmidrule(lr){7-7}"
        + "\n"
    )

    f.write(
        r"& G2S & LASSO & G2S & LASSO & G2S & G2S \\"
        + "\n"
    )

    f.write(r"\cmidrule{2-7}" + "\n")

    ##########################################################################
    # BODY
    ##########################################################################

    for line in latex_lines_exog:
        f.write(line + "\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabularx}" + "\n")
    f.write(r"\end{table}" + "\n")

print("\nSaved exogenous model LaTeX table:")
print(latex_file_exog)

##############################################################################
##############################################################################
# TABLES WITH SELECTED VARIABLES
##############################################################################
##############################################################################

##############################################################################
# VARIABLE LABELS
##############################################################################

X_labels = [
    'RB', 'WPAC', 'WPAC2', 'WPAC3', 'WPAC4', '50fin',
    '100cold', '100fin1', '100fin2', '150fin1', '150fin2',
    '200fin1', '200fin2', '250fin1', '250fin2', '300fin1',
    '300fin2', '400fin1', '400fin2', '500fin1', '500fin2',
    'wnd160.200_0.10', 'wnd180.220_-4.4', 'wnd180.210_-10.0'
]

##############################################################################
# PRETTY LABELS
##############################################################################

pretty_names = {
    'RB': 'RB',
    'WPAC': 'WPAC',
    'WPAC2': 'WPAC2',
    'WPAC3': 'WPAC3',
    'WPAC4': 'WPAC4',

    '50fin': '50m',
    '100cold': '100m "cold"',
    '100fin1': '100m Region 1',
    '100fin2': '100m Region 2',

    '150fin1': '150m Region 1',
    '150fin2': '150m Region 2',

    '200fin1': '200m Region 1',
    '200fin2': '200m Region 2',

    '250fin1': '250m Region 1',
    '250fin2': '250m Region 2',

    '300fin1': '300m Region 1',
    '300fin2': '300m Region 2',

    '400fin1': '400m Region 1',
    '400fin2': '400m Region 2',

    '500fin1': '500m Region 1',
    '500fin2': '500m Region 2',

    'wnd160.200_0.10': 'WND Region 3',
    'wnd180.220_-4.4': 'WND Region 1',
    'wnd180.210_-10.0': 'WND Region 2',
}

##############################################################################
# READ SELECTED LAGS
##############################################################################

def get_selected_lags(file_path, variables):

    df = pd.read_csv(file_path)

    selected = {}

    for var in variables:
        selected[var] = []

    for _, row in df.iterrows():

        param = str(row["Param"]).strip()

        if "_" not in param:
            continue

        try:
            coef = float(row["Est"])
        except:
            continue

        if coef == 0:
            continue

        split = param.rsplit("_", 1)

        if len(split) != 2:
            continue

        var_name = split[0]

        try:
            lag = int(split[1])
        except:
            continue

        if var_name in selected:
            selected[var_name].append(lag)

    for var in selected:
        selected[var] = sorted(selected[var])

    return selected

##############################################################################
# BUILD CONFIGURATIONS
##############################################################################
configs_no_pca = []

for horizon in horizons:

    configs_no_pca.append({
        "label": (horizon, "LASSO"),
        "include_X": True,
        "method": "Lasso",
        "num_comps": 0,
        "setting": "A",
        "bool_use_PCA": False,
        "horizon": horizon,
    })

    configs_no_pca.append({
        "label": (horizon, "G2S_constant"),
        "include_X": True,
        "method": "G2S",
        "num_comps": 0,
        "setting": "A",
        "bool_use_PCA": False,
        "horizon": horizon,
    })

    configs_no_pca.append({
        "label": (horizon, "G2S_cycles"),
        "include_X": True,
        "method": "G2S",
        "num_comps": 5,
        "setting": "B",
        "bool_use_PCA": False,
        "horizon": horizon,
    })

##############################################################################
# PCA CONFIGS
##############################################################################

configs_pca = []

for horizon in horizons:

    configs_pca.append({
        "label": (horizon, "LASSO"),
        "include_X": True,
        "method": "Lasso",
        "num_comps": 0,
        "setting": "A",
        "bool_use_PCA": True,
        "horizon": horizon,
    })

    configs_pca.append({
        "label": (horizon, "G2S_constant"),
        "include_X": True,
        "method": "G2S",
        "num_comps": 0,
        "setting": "A",
        "bool_use_PCA": True,
        "horizon": horizon,
    })

    configs_pca.append({
        "label": (horizon, "G2S_cycles"),
        "include_X": True,
        "method": "G2S",
        "num_comps": 5,
        "setting": "B",
        "bool_use_PCA": True,
        "horizon": horizon,
    })

##############################################################################
# LOAD ALL RESULTS
##############################################################################

def load_results(configs, variables):

    out = {}

    for config in configs:
        config["ar_lags"] = 0
        config["bool_use_seas_dummies"] = False
        key = config["label"]

        string_info = build_string_info(config)

        file_path = os.path.join(
            parameter_folder,
            f"params{string_info}.csv"
        )

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        out[key] = get_selected_lags(
            file_path,
            variables
        )

    return out


##############################################################################
# NON-PCA TABLES
##############################################################################

non_pca_results = load_results(
    configs_no_pca,
    X_labels
)

##############################################################################
# PCA TABLES
##############################################################################

pca_variables = [f"PC{i}" for i in range(1, 7)]

pca_results = load_results(
    configs_pca,
    pca_variables
)

##############################################################################
# HELPER
##############################################################################

def lags_to_string(lags):

    if len(lags) == 0:
        return ""

    return ", ".join(str(x) for x in lags)

##############################################################################
# WRITE NON-PCA TABLE
##############################################################################

def write_non_pca_table(horizons_subset, filename):

    file_path = os.path.join(output_folder, filename)

    with open(file_path, "w") as f:

        f.write(r"\begin{table}[h!]" + "\n")
        f.write(r"\footnotesize" + "\n")
        f.write(r"\centering" + "\n")

        f.write(
            r"\begin{tabularx}{\textwidth}{l LLL LLL}"
            + "\n"
        )

        f.write(r"\toprule" + "\n")

        h1 = horizons_subset[0]
        h2 = horizons_subset[1]

        f.write(
            rf"& \multicolumn{{3}}{{c}}{{horizon {h1}}}"
            rf"& \multicolumn{{3}}{{c}}{{horizon {h2}}} \\"
            + "\n"
        )

        f.write(
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}"
            + "\n"
        )

        f.write(
            r"& \multicolumn{2}{c}{constant}"
            r"& \multicolumn{1}{c}{seas + cycles}"
            r"& \multicolumn{2}{c}{constant}"
            r"& \multicolumn{1}{c}{seas + cycles} \\"
            + "\n"
        )

        f.write(
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-4}"
            r"\cmidrule(lr){5-6} \cmidrule(lr){7-7}"
            + "\n"
        )

        f.write(
            r"& \multicolumn{1}{c}{LASSO}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{LASSO}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{G2S} \\"
            + "\n"
        )

        f.write(r"\cmidrule{2-7}" + "\n")

        for var in X_labels:

            row = [pretty_names[var]]

            for horizon in horizons_subset:

                for model in [
                    "LASSO",
                    "G2S_constant",
                    "G2S_cycles",
                ]:

                    key = (horizon, model)

                    if key not in non_pca_results:
                        row.append("")
                        continue

                    lags = non_pca_results[key][var]

                    row.append(
                        lags_to_string(lags)
                    )

            f.write(
                " & ".join(row)
                + r" \\"
                + "\n"
            )

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabularx}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"Saved: {file_path}")

##############################################################################
# WRITE PCA TABLE
##############################################################################

def write_pca_table(file_name):

    file_path = os.path.join(
        output_folder,
        file_name
    )

    with open(file_path, "w") as f:

        f.write(r"\begin{table}[h!]" + "\n")
        f.write(r"\footnotesize" + "\n")
        f.write(r"\centering" + "\n")

        f.write(
            r"\begin{tabularx}{\textwidth}{l LLL LLL LLL LLL}"
            + "\n"
        )

        f.write(r"\toprule" + "\n")

        f.write(
            r"& \multicolumn{3}{c}{horizon 1}"
            r"& \multicolumn{3}{c}{horizon 6}"
            r"& \multicolumn{3}{c}{horizon 12}"
            r"& \multicolumn{3}{c}{horizon 18} \\"
            + "\n"
        )

        f.write(
            r"\cmidrule(lr){2-4}"
            r"\cmidrule(lr){5-7}"
            r"\cmidrule(lr){8-10}"
            r"\cmidrule(lr){11-13}"
            + "\n"
        )

        f.write(
            r"& \multicolumn{2}{c}{constant}"
            r"& \multicolumn{1}{c}{seas + cycles}"
            r"& \multicolumn{2}{c}{constant}"
            r"& \multicolumn{1}{c}{seas + cycles}"
            r"& \multicolumn{2}{c}{constant}"
            r"& \multicolumn{1}{c}{seas + cycles}"
            r"& \multicolumn{2}{c}{constant}"
            r"& \multicolumn{1}{c}{seas + cycles} \\"
            + "\n"
        )

        f.write(
            r"\cmidrule(lr){2-3}"
            r"\cmidrule(lr){4-4}"
            r"\cmidrule(lr){5-6}"
            r"\cmidrule(lr){7-7}"
            r"\cmidrule(lr){8-9}"
            r"\cmidrule(lr){10-10}"
            r"\cmidrule(lr){11-12}"
            r"\cmidrule(lr){13-13}"
            + "\n"
        )

        f.write(
            r"& \multicolumn{1}{c}{LASSO}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{LASSO}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{LASSO}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{LASSO}"
            r"& \multicolumn{1}{c}{G2S}"
            r"& \multicolumn{1}{c}{G2S} \\"
            + "\n"
        )

        f.write(r"\cmidrule{2-13}" + "\n")

        for var in pca_variables:

            row = [var.replace("PC", "PC ")]

            for horizon in horizons:

                for model in [
                    "LASSO",
                    "G2S_constant",
                    "G2S_cycles",
                ]:

                    key = (horizon, model)

                    if key not in pca_results:
                        row.append("")
                        continue

                    lags = pca_results[key][var]

                    row.append(
                        lags_to_string(lags)
                    )

            f.write(
                " & ".join(row)
                + r" \\"
                + "\n"
            )

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabularx}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"Saved: {file_path}")

##############################################################################
# CREATE TABLES
##############################################################################

write_non_pca_table(
    [1, 6],
    table_names[3]+".tex"
)

write_non_pca_table(
    [12, 18],
    table_names[4]+".tex"
)

write_pca_table(table_names[5]+".tex")
