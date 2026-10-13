# ======================================================================================================================
"""
This script computes MSE-values corresponding to Tables 3, A.8, A.9, A.10, and A.11.
Note: the MSE-values in Table A.11 are a subset of those in Table 3.

Dependencies:
- pandas
- numpy

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

import config as cfg


#%% MSE tables
events = ['Main', 'Apdx', 'Nino', 'Nina']

models = {
    "Main": cfg.Main,
    "Apdx": cfg.Appendix,
    "Nino": cfg.Main,
    "Nina": cfg.Main,
}

event_periods = {
    "Main": cfg.test_sample,
    "Apdx": cfg.test_sample,
    "Nino": cfg.Nino,
    "Nina": cfg.Nina,
}

titles = {
    "Main": "Table 3 and Table A.11",
    "Apdx": "Table A.8",
    "Nino": "Table A.9",
    "Nina": "Table A.10",
}

for event in events:
    L2 = {}

    for h in cfg.horizons:
        file_name = '../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon' + str(h) + '.csv'
        tmp = pd.read_csv(file_name, index_col=0)**2

        # Extract time periods corresponding to event
        event_match = [tmp.index[i][:-3] in event_periods[event] for i in range(len(tmp.index))]

        # Remove horizon-specifc suffix from variable names
        tmp.columns = [col.replace(f"horizon{h}", "") for col in tmp.columns]

        # Extract models for the main text/appendix
        tmp = tmp[models[event]]

        for model in tmp.columns:
            if h == 1:
                L2[model] = np.zeros(len(cfg.horizons))

            L2[model][cfg.horizons.index(h)] = tmp[model].values[event_match].mean()

    # Storing output as a DataFrame
    df = pd.DataFrame(L2).T
    df.columns = cfg.horizons
    df = df.rename(index=cfg.rename_map)

    # Creating LaTeX output
    latex = df.to_latex(
        escape=True,
        float_format="%.3f".__mod__,
        bold_rows=False,
    )

    print(titles[event])
    print(latex)







