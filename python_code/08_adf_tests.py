# ======================================================================================================================
"""
This script computes augmented Dickey-Fuller tests based on the pairwise forecast loss differentials underlying
Tables 3, A.8, A.9, A.10, and A.11

Dependencies:
- pandas
- numpy
- statsmodels

By Sebastian Jensen
Aarhus University
CoRE
Jul, 2026
"""
# ======================================================================================================================

#%% Python environment
import os; os.environ['PYTHONHASHSEED'] = str(0)
import sys; sys.path.append('python_code')

import numpy as np
import pandas as pd
import itertools

from statsmodels.tsa.stattools import adfuller

import config as cfg


#%% ADF p-values
events = ['Main', 'Apdx', 'Nino', 'Nina', 'no_anoms']

models = {
    "Main": cfg.Main,
    "Apdx": cfg.Appendix,
    "Nino": cfg.Main,
    "Nina": cfg.Main,
    "no_anoms": cfg.Score_nams,
}

event_periods = {
    "Main": cfg.test_sample,
    "Apdx": cfg.test_sample,
    "Nino": cfg.Nino,
    "Nina": cfg.Nina,
    "no_anoms": cfg.test_sample,
}

pvals = {}

for event in events:
    for h in cfg.horizons:
        file_name = '../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon' + str(h) + '.csv'
        L = pd.read_csv(file_name, index_col=0) ** 2

        # Extract time periods corresponding to event
        event_match = [L.index[i][:-3] in event_periods[event] for i in range(len(L.index))]
        L = L.iloc[event_match]

        # Remove horizon-specifc suffix from variable names
        L.columns = [col.replace(f"horizon{h}", "") for col in L.columns]

        # Extract models for the main text/appendix
        L = L[models[event]]

        if h == 1:
            combs = list(itertools.combinations(L.columns, 2))
            pvals[event] = np.zeros((len(cfg.horizons), len(combs)))

        for i in range(len(combs)):
            d = L[combs[i][0]] - L[combs[i][1]]

            adf_pval = adfuller(d, regression='c', autolag='BIC')[1]

            pvals[event][cfg.horizons.index(h), i] = adf_pval


#%% Printing results
print("Percentage of failed ADF rejections on pairwise forecast loss differentials")
print("-" * 70)

results = {
    "Full test sample (Table 3)": pvals["Main"],
    "Seasonal cycles vs. seasonal dummies (Table A.8)": pvals["Apdx"],
    "Niño sub-sample (Table A.9)": pvals["Nino"],
    "Niña sub-sample (Table A.10)": pvals["Nina"],
    "No-anomalies benchmarks (Table A.11)": pvals["no_anoms"],
}

for label, vals in results.items():
    frac = np.mean(vals > 0.05)
    print(f"{label:<50} {frac:.1%}")

