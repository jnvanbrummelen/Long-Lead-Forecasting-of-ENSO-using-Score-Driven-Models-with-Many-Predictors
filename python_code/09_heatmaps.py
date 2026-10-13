# ======================================================================================================================
"""
This script plots heatmaps of the mean squared forecast error (MSE) to investigate the ENSO spring predictability barrier;
it is used to create Figures 4 and 5.

Dependencies:
- pandas
- numpy
- seaborn
- matplotlib

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

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False, "axes.spines.bottom": False}
import seaborn as sns; sns.set_theme(style='white', rc=custom_params)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

import config as cfg


#%% Loading data
L2 = {}

for h in cfg.horizons:
    file_name = '../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon' + str(h) + '.csv'
    tmp = pd.read_csv(file_name, index_col=0)**2

    # remove horizon-specifc suffix
    tmp.columns = [col.replace(f"horizon{h}", "") for col in tmp.columns]

    month_list = [tmp.index[i].rsplit('-', 2)[1] for i in range(len(tmp.index))]

    for month in range(1, 12 + 1):
        month_match = [month_list[i] == f"{month:02d}" for i in range(len(month_list))]

        for model in tmp.columns:
            if h == 1 and month == 1:
                L2[model] = np.zeros((len(cfg.horizons), 12))

            L2[model][cfg.horizons.index(h), month - 1] = tmp[model].values[month_match].mean()


#%% Figure setup
models = ['Lasso0componentsSettingAwithExog', 'G2S0componentsSettingAwithExogPCA', '5componentsSettingB',
          'G2S5componentsSettingBwithExogPCA', '0componentsSettingA_AR1_anomalies', 'G2S0componentsSettingA_AR1_anomalieswithExog',
          '0componentsSettingA_AR_anomalies', 'G2S0componentsSettingA_AR_anomalieswithExogPCA']

names = ['Regressors: LASSO', 'Regressors: G2S PCA', 'Constant + seasonals + cycles', 'Constant + seasonals + cycles + G2S PCA regressors',
         'Anomalies: AR(1)', 'Anomalies: ARX(1) G2S', 'Anomalies: AR(3)', 'Anomalies: ARX(3) G2S PCA']

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
mmonths = months + months
mmmonths = months + months + months

pos = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]

# Colorbar limits
all_vals = np.concatenate([np.asarray(L2[m]).ravel() for m in models])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

norm = colors.Normalize(vmin=vmin, vmax=vmax)


#%% Heatmap as a function of the initialization month: Figure 4
fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(12,13))
fig.tight_layout()

for k in range(len(models)):
    model = models[k]

    L2_tmp = np.zeros(L2[model].shape)

    for i in range(len(cfg.horizons)):
        if cfg.horizons[i] <= 12:
            L2_tmp[i, :] = np.concatenate((L2[model][i, cfg.horizons[i]:], L2[model][i, :cfg.horizons[i]]))
        else:
            L2_tmp[i, :] = np.concatenate((L2[model][i, cfg.horizons[i] - 12:], L2[model][i, :cfg.horizons[i] - 12]))

        for j in range(len(months)):
            text1 = axs[pos[k]].text(j, i, '\n\n' + mmmonths[j+cfg.horizons[i]], ha="center", va="center", color="w", fontsize=8)
            text2 = axs[pos[k]].text(j, i, np.round(L2_tmp[i, j], 2), ha="center", va="center", color="w")

    im = axs[pos[k]].imshow(L2_tmp, norm=norm)

    if k in [1,3,5,7]:
        divider = make_axes_locatable(axs[pos[k]])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = axs[pos[k]].figure.colorbar(im, ax=axs[pos[k]], cax=cax)
        cbar.ax.set_ylabel('MSE', rotation=-90, va="bottom")
        cbar.outline.set_visible(False)

    # Show all ticks and label them with the respective list entries
    axs[pos[k]].set_xticks(np.arange(len(months)))
    axs[pos[k]].set_yticks(np.arange(len(cfg.horizons)))

    axs[pos[k]].set_xticklabels(months)
    axs[pos[k]].set_yticklabels(cfg.horizons)

    axs[pos[k]].set_xticks(np.arange(L2[model].shape[1] + 1) - .5, minor=True)
    axs[pos[k]].set_yticks(np.arange(L2[model].shape[0] + 1) - .5, minor=True)
    axs[pos[k]].grid(which="minor", color="w", linestyle='-', linewidth=1)
    axs[pos[k]].tick_params(which="minor", bottom=False, left=False)

    axs[pos[k]].set_title(names[k])

    if k in [6, 7]:
        axs[pos[k]].set_xlabel('Initialization month')

    if k in [0, 2, 4, 6]:
        axs[pos[k]].set_ylabel('Forecasting horizon')

    # Rotate the tick labels and set their alignment.
    plt.setp(axs[pos[k]].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.savefig('../output/figures/Figure4_heat_initialization_month', bbox_inches='tight')
plt.show()


#%% Heatmap as a function of the target month: Figure 5
fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(12,13))
fig.tight_layout()

for k in range(len(models)):
    model = models[k]

    im = axs[pos[k]].imshow(L2[model], norm=norm)

    if k in [1,3,5,7]:
        divider = make_axes_locatable(axs[pos[k]])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = axs[pos[k]].figure.colorbar(im, ax=axs[pos[k]], cax=cax)
        cbar.ax.set_ylabel('MSE', rotation=-90, va="bottom")
        cbar.outline.set_visible(False)

    # Show all ticks and label them with the respective list entries
    axs[pos[k]].set_xticks(np.arange(len(months)))
    axs[pos[k]].set_yticks(np.arange(len(cfg.horizons)))

    axs[pos[k]].set_xticklabels(months)
    axs[pos[k]].set_yticklabels(cfg.horizons)

    for i in range(len(cfg.horizons)):
        for j in range(len(months)):
            text1 = axs[pos[k]].text(j, i, '\n\n' + mmonths[j-cfg.horizons[i]], ha="center", va="center", color="w", fontsize=8)
            text2 = axs[pos[k]].text(j, i, np.round(L2[model][i, j], 2), ha="center", va="center", color="w")

    axs[pos[k]].set_xticks(np.arange(L2[model].shape[1] + 1) - .5, minor=True)
    axs[pos[k]].set_yticks(np.arange(L2[model].shape[0] + 1) - .5, minor=True)
    axs[pos[k]].grid(which="minor", color="w", linestyle='-', linewidth=1)
    axs[pos[k]].tick_params(which="minor", bottom=False, left=False)

    axs[pos[k]].set_title(names[k])

    if k in [6,7]:
        axs[pos[k]].set_xlabel('Target month')

    if k in [0,2,4,6]:
        axs[pos[k]].set_ylabel('Forecasting horizon')

    # Rotate the tick labels and set their alignment.
    plt.setp(axs[pos[k]].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.savefig('../output/figures/Figure5_heat_target_month', bbox_inches='tight')
plt.show()
