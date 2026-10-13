#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
""" 
@author: Janneke van Brummelen
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

from python_code.methods.EN_model_class import *
import matplotlib.dates as mdates
import datetime as dt

###########################################################
### StringToDate(rawStringDates)
def StringToDate(rawStringDates):
    """
    Purpose:
        Convert string to datetime.date object

    Inputs:
        vIntDate    vector, dates as strings (example: 2016-06-18)

    Return value:
        vNumDates   vector, dates converted to days since 01-01-01
        vsStringDates   vector, reformatted date strings in ymd format
    """
    N = np.size(rawStringDates)
    num_dates = np.zeros((N, 1))
    string_dates = []

    for i in range(N):
        date = rawStringDates[i]
        year = int(date[:4])
        month = int(date[5:])
        DateTime = dt.date(year, month, 1)
        num_dates[i] = mdates.date2num(DateTime)
        string_dates.append(date[:4] + "-" + date[5:] + "-01")

    return num_dates, np.array(string_dates)

###########################################################
### save_results(sdcm, string_info)
def save_results(sdcm,output_folder, string_info, verbose,string_dates=None):
    if string_dates is None:
        string_dates = np.arange(sdcm.y.shape[0])
    df_tot = sdcm.summarize_results(print_results=verbose, bool_plot_filt_vals=False)

    paths = sdcm.paths[sdcm.burn:,:]
    if sdcm.num_comps > 1 and not sdcm.bool_trend:
        sdcm.paths[:,0] = sdcm.params.omega
    df_filt = pd.DataFrame(np.hstack((sdcm.y[sdcm.burn:].reshape(-1,1), paths)),index=string_dates.reshape(-1,),columns=['data']+['f hat '+str(i+1) for i in range(sdcm.num_comps)])
    with open(output_folder+'/paths/FilteredSeqs'+string_info+'.csv','w') as tf:
        tf.write(df_filt.to_csv())  

    with open(output_folder+'/parameters/params'+string_info+'.csv','w') as tf:
        tf.write(df_tot.to_csv())  
         
    return df_tot

def Parameter_estimation(output_folder, Y_vec, X_vars,num_dates,string_dates,X_lags, X_labels, bool_use_random_search, bool_use_dates, sig_level, burn, num_avg,
                         max_lag_UB, skip_first, horizon, include_X, bool_use_anomalies, bool_use_seas_dummies, bool_use_PCA,
                         ar_lags, num_comps, setting, method,insample_length=None,save_res=True):
    np.random.seed(1234)
    lambda_grid = np.logspace(-2.1, 1.2, 100) # grid of penalty parameters
    # randomization configuration
    n_draws= 10000
    n_best= 5
    verbose = False

    if ar_lags>0:
        bool_use_AR_lags = True
        setting = 'A'
    else:
        bool_use_AR_lags = False

    init_params = {}
    param_restrictions = {}
    if setting == 'A':
        num_comps = 0
    elif setting == 'B':
        param_restrictions={'phi_1':'fix_all','theta_1':'fix_all'}
        init_params = {'phi_1':np.array([0]),'theta_1':np.array([0])}
    elif setting == 'C':
        param_restrictions={'omega':'fix_all','phi_1':'fix_all'}
        init_params = {'omega': 0.0, 'phi_1':np.array([1.0])}

    vec_autoreg_lags = np.array([1]+[2 for i in range(num_comps-1)])
    vec_ma_lags = np.array([1]+[1 for i in range(num_comps-1)])

    if bool_use_PCA:
        num_PCA = 6
    else:
        num_PCA = 0

    # periodicity restrictions
    cycle_per_restrictions = {}
    if num_comps==3 and not bool_use_seas_dummies:
        cycle_per_restrictions={2:(6,6),3:(12,12)}
    elif num_comps==3:
        cycle_per_restrictions={2:(13,21),3:(23,43)}
        init_params.update({'phi_2':np.array([1.1,-0.032]),'phi_3':np.array([1.1,-0.691])})
    elif num_comps==5:
        cycle_per_restrictions={2:(6,6),3:(12,12),4:(13,21),5:(23,43)}

    # starting values
    if not setting == 'C':
        init_params['omega'] = 26.0
    init_params['sigma'] = 0.1
    if num_comps == 5:
        init_params.update({'phi_2':np.array([0.986,-0.984]),'phi_3':np.array([1.1,-0.996]),'phi_4':np.array([1.1,-0.032]),'phi_5':np.array([1.1,-0.691])})

    if bool_use_seas_dummies:
        init_params.update({'beta_seas':np.array([-0.09,0.1,0.5,1.0,1.1,0.9,0.6,0.2,0.14,0.06,0.8])})

    if skip_first == 240 and num_comps == 5:
        cycle_per_restrictions = {2: (6, 6), 3: (12, 12), 5: (23, 43)}
        param_restrictions['phi_4'] = 'fix_all'
        init_params.update({'phi_4':np.array([0,0])})

    for key, value in cycle_per_restrictions.items():
        if value[0] == value[1]:
            param_restrictions['phi_' + str(key)] = 'fix_frequency'

    T = Y_vec.shape[0]

    #########################################################################
    ########################### ESTIMATION ##################################
    #########################################################################
    sdcm_noexog = ScoreDrivenComponentModel(Y_vec.reshape(-1,),dates=num_dates,T_insample=insample_length,num_comps=num_comps,bool_include_AR=bool_use_AR_lags,
           vec_autoreg_lags=vec_autoreg_lags, vec_ma_lags=vec_ma_lags, param_restrictions=param_restrictions,
           init_params=init_params, burn=burn, cycle_per_restrictions=cycle_per_restrictions,num_avg=num_avg,
           horizon=horizon, relevant_X_lags=X_lags[:bool_use_AR_lags],bool_seas_dummies=bool_use_seas_dummies)
    sdcm_noexog.estimate(print_results=verbose,check_hess_posdef=False)
    if bool_use_random_search:
        sdcm_noexog_rs = ScoreDrivenComponentModel(Y_vec.reshape(-1,),dates=num_dates,T_insample=insample_length,num_comps=num_comps,bool_include_AR=bool_use_AR_lags,
           vec_autoreg_lags=vec_autoreg_lags, vec_ma_lags=vec_ma_lags, param_restrictions=param_restrictions,
           init_params=init_params, burn=burn, cycle_per_restrictions=cycle_per_restrictions,num_avg=num_avg,
           horizon=horizon, relevant_X_lags=X_lags[:bool_use_AR_lags],bool_seas_dummies=bool_use_seas_dummies)
        sdcm_noexog_rs.random_start_search(construct_std_errors=True, print_progress=verbose,
                             n_draws=n_draws, n_best=n_best, check_hess_posdef=False)
        if sdcm_noexog_rs.llh > sdcm_noexog.llh or np.isnan(sdcm_noexog.llh):
            sdcm_noexog = sdcm_noexog_rs

    if include_X:
        if method == "G2S":
            sdcm_g2s = ScoreDrivenComponentModel(Y_vec.reshape(-1,), X=X_vars,T_insample=insample_length,dates=num_dates,num_comps=num_comps,bool_include_AR=bool_use_AR_lags,
                   vec_autoreg_lags=vec_autoreg_lags, vec_ma_lags=vec_ma_lags, param_restrictions=param_restrictions,
                   init_params=init_params, burn=burn, cycle_per_restrictions=cycle_per_restrictions,num_avg=num_avg,
                   relevant_X_lags=X_lags, X_labels=X_labels,horizon=horizon,num_PCA=num_PCA, bool_seas_dummies=bool_use_seas_dummies)
            flat_no_exog = sdcm_noexog._params_to_vector(sdcm_noexog.params)
            flat_g2s = sdcm_g2s._params_to_vector(sdcm_g2s.params)
            flat_g2s[:flat_no_exog.shape[0]] = flat_no_exog
            sdcm_g2s.params = sdcm_g2s._params_from_vector(flat_g2s)
            sdcm_g2s.general_to_specific_grouped(sig_level=sig_level,bool_pre_estimate=False,bool_random_search=bool_use_random_search,print_progress=verbose)
        elif num_comps == 0 and method == "Lasso":
            sdcm_lasso = ScoreDrivenComponentModel(Y_vec.reshape(-1,), T_insample=insample_length,X=X_vars,dates=num_dates,num_comps=num_comps, bool_include_AR=bool_use_AR_lags,
                                             param_restrictions=param_restrictions, init_params=init_params,
                                             burn=burn, relevant_X_lags=X_lags,
                                             X_labels=X_labels,horizon=horizon,num_PCA=num_PCA,bool_seas_dummies=bool_use_seas_dummies)
            sdcm_lasso.estimate_lasso_gridsearch(lambda_grid,print_results=verbose,print_subresults=verbose)


    ### Save results
    string_info = str(num_comps)+'componentsSetting'+setting+ bool_use_AR_lags*'_AR'+bool_use_seas_dummies*'_seasdummies'+bool_use_anomalies*'_anomalies'+('_horizon'+str(horizon))*include_X
    if skip_first > 0:
        string_info += 'skipfirst'+str(skip_first)
    if save_res:
        df_res = save_results(sdcm_noexog,output_folder,string_info,verbose,string_dates=string_dates[max_lag_UB+burn:])
    if include_X:
        string_info = string_info + 'withExog' + bool_use_PCA * 'PCA'
        if method == "G2S":
            string_info = string_info + '_G2S'
            if save_res:
                save_results(sdcm_g2s,output_folder,string_info,verbose,string_dates=string_dates[max_lag_UB+burn:])
        elif num_comps == 0 and method == "Lasso":
            string_info = string_info + '_Lasso'
            if save_res:
                save_results(sdcm_lasso,output_folder,string_info,verbose,string_dates=string_dates[max_lag_UB+burn:])

    print(string_info)

    string_info = (method * include_X + str(num_comps) + 'componentsSetting' + setting
                   + bool_use_seas_dummies * '_seasdummies' + bool_use_anomalies * '_anomalies' + 'withExog' * include_X
                   + bool_use_PCA * 'PCA')
    if method == "G2S":
        sdcm_fin = sdcm_g2s
    elif method == "Lasso":
        sdcm_fin = sdcm_lasso
    else:
        sdcm_fin = sdcm_noexog
    return sdcm_fin, string_info