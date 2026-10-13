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
from joblib import Parallel, delayed

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
    num_dates = np.zeros((N,1))
    string_dates = []
    
    for i in range(N):
        date = rawStringDates[i]  
        year =  int(date[:4])  
        month = int(date[5:])  
        DateTime = dt.date(year, month, 1)
        num_dates[i] = mdates.date2num(DateTime)  
        string_dates.append(date[:4]+"-"+date[5:]+"-01")

    return num_dates, np.array(string_dates)

###########################################################
### save_forecast_results(param_labels,dates,param_ests,forecast_vec,forecast_error_vec,string_info,bool_use_pre_estim)
def save_forecast_results(output_folder,param_labels,dates,param_ests,forecast_vec,forecast_erorr_vec,string_info,bool_use_pre_estim):
    if not bool_use_pre_estim:
        df_params = pd.DataFrame(param_ests,index=param_labels)
        with open(output_folder+'parameter_estimates/params'+string_info+'.csv','w') as tf:
            tf.write(df_params.to_csv())

    if forecast_vec.shape[0] > 1:
        df_forecasts =  pd.DataFrame(forecast_vec,index=dates.reshape(-1,))
        with open(output_folder+'forecasts/forecasts'+string_info+'.csv','w') as tf:
            tf.write(df_forecasts.to_csv())

        df_forecast_errors =  pd.DataFrame(forecast_erorr_vec,index=dates.reshape(-1,))
        with open(output_folder+'forecast_errors/forecast_errors'+string_info+'.csv','w') as tf:
            tf.write(df_forecast_errors.to_csv())


def EstimationAndForecast(output_folder,insample_length, Y_vec, X_vars,num_dates,string_dates,X_lags, X_labels, use_random_search, bool_use_dates, sig_level, burn, num_avg,
                         max_lag_UB, skip_first, horizon, include_X, bool_use_anomalies, bool_use_seas_dummies, bool_use_PCA,
                         ar_lags, num_comps, setting, method, bool_use_pre_estim, bool_use_pre_estim_noexog,n_jobs=1,save_results=True):

    np.random.seed(1234)
    lambda_grid = np.logspace(-2.1, 1.2, 100)

    n_draws=5000
    n_best=5
    verbose = False
    if bool_use_pre_estim or (not bool_use_pre_estim_noexog and (num_comps > 0 or not include_X)):
        n_jobs=1
    bool_parallelize = (n_jobs > 1)

    if ar_lags>0:
        bool_use_AR_lags = True
        setting = 'A'
    else:
        bool_use_AR_lags = False

    init_params = {}
    param_restrictions = {}
    if setting == 'A':
        num_comps = 0
    elif setting == 'B': # constant, no trend
        param_restrictions={'phi_1':'fix_all','theta_1':'fix_all'}
        init_params = {'phi_1':np.array([0.0]),'theta_1':np.array([0.0])}
    elif setting == 'C': # no constant, trend
        param_restrictions={'omega':'fix_all','phi_1':'fix_all'}
        init_params = {'omega': 0.0, 'phi_1':np.array([1.0])}

    if num_comps == 0:
        use_random_search = False

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
        cycle_per_restrictions = {2: (6,6), 3: (12,12), 4: (13, 21), 5: (23, 43)}

    # starting values
    if not setting == 'C':
        init_params['omega'] = 26.0
    init_params['sigma'] = 0.1
    if num_comps == 5:
        init_params.update({'phi_2':np.array([0.986,-0.984]),'phi_3':np.array([1.1,-0.996]),'phi_4':np.array([1.1,-0.032]),'phi_5':np.array([1.1,-0.691])})

    if bool_use_seas_dummies:
        init_params.update({'beta_seas':np.array([-0.09,0.1,0.5,1.0,1.1,0.9,0.6,0.2,0.14,0.06,0.8])})

    outsample_length = Y_vec.shape[0] - insample_length

    #########################################################################
    ########################### ESTIMATION ##################################
    #########################################################################
    def _make_model(Y, X=None, start_month=0, relevant_X_lags=None, X_labels=None, horizon=1, num_PCA=0):
        return ScoreDrivenComponentModel(
            Y.reshape(-1,), X=X, T_insample=insample_length,
            num_comps=num_comps, bool_include_AR=bool_use_AR_lags,
            vec_autoreg_lags=vec_autoreg_lags, vec_ma_lags=vec_ma_lags,
            param_restrictions=param_restrictions, init_params=init_params,
            burn=burn, cycle_per_restrictions=cycle_per_restrictions,
            num_avg=num_avg, horizon=horizon, num_PCA=num_PCA,
            bool_seas_dummies=bool_use_seas_dummies,
            start_month=start_month,
            relevant_X_lags=relevant_X_lags,
            X_labels=X_labels
        )

    outsample_length_hor = outsample_length - horizon + 1

    if horizon == 1 and outsample_length == 240:
        Y_vec_hor = Y_vec[:-1]
        X_vars_hor = X_vars[:-1, :]
        string_dates_hor = string_dates[:-1]
        outsample_length_hor = outsample_length_hor - 1
    else:
        Y_vec_hor = Y_vec
        X_vars_hor = X_vars
        string_dates_hor = string_dates

    sdcm_dummy = _make_model(Y_vec, relevant_X_lags=X_lags[:bool_use_AR_lags], horizon=horizon)
    param_ests_noexog = np.zeros((sdcm_dummy.get_param_vector().shape[0], outsample_length_hor))
    if include_X:
        sdcm_dummy = _make_model(Y_vec_hor, X=X_vars_hor, relevant_X_lags=X_lags, X_labels=X_labels, horizon=horizon, num_PCA=num_PCA)
        param_ests = np.zeros((sdcm_dummy.get_param_vector().shape[0], outsample_length_hor))
    else:
        param_ests = param_ests_noexog

    forecast_vec = np.zeros((outsample_length_hor,))
    forecast_error_vec = np.zeros((outsample_length_hor,))

    if bool_use_pre_estim:
        if not include_X:
            alt_hor = 1
        else:
            alt_hor = horizon
        str_ar_lags = '' if (ar_lags == 3 or ar_lags == 0) else str(ar_lags)
        alt_string_info = method * include_X + str(
            num_comps) + 'componentsSetting' + setting + bool_use_AR_lags * (
                                      '_AR' + str_ar_lags) + bool_use_seas_dummies * '_seasdummies' + bool_use_anomalies * '_anomalies' + 'withExog' * include_X + bool_use_PCA * 'PCA' + 'horizon' + str(
            alt_hor)
        df_params = pd.read_csv(output_folder+'parameter_estimates/params' + alt_string_info + '.csv', header=0)
        params = np.array(df_params.values[:, 1:], dtype=np.float64)
    elif bool_use_pre_estim_noexog:
        alt_string_info = str(num_comps) + 'componentsSetting' + setting + bool_use_seas_dummies * '_seasdummies' + 'horizon1'
        df_nonexog_params = pd.read_csv(output_folder + 'parameter_estimates/params' + alt_string_info + '.csv', header=0)
        param_ests_noexog = np.array(df_nonexog_params.values[:, 1:], dtype=np.float64)

    def run_single_estimation_and_forecast(i):
        #start = timeit.default_timer()
        Y_i = Y_vec_hor[i:]
        X_i = X_vars_hor[i:, :]
        start_month_i = i % 12

        if bool_use_pre_estim:
            sdcm_fin = _make_model(Y_i, X=X_i, start_month=start_month_i,
                                   relevant_X_lags=X_lags[:bool_use_AR_lags + include_X * len(X_labels)],
                                   X_labels=X_labels[:include_X * len(X_labels)], horizon=horizon, num_PCA=num_PCA)
            sdcm_fin.params = sdcm_fin._params_from_vector(params[:, i])
            param_ests_vec = params[:, i]
        else:
            if method != 'Lasso':
                if not bool_use_pre_estim_noexog and (num_comps > 0 or not include_X):
                    sdcm_noexog = _make_model(Y_i, start_month=start_month_i, relevant_X_lags=X_lags[:bool_use_AR_lags],
                                              horizon=horizon)
                    sdcm_noexog.estimate(print_results=verbose, construct_std_errors=False)
                    if i > 0:
                        sdcm_noexog_alt = _make_model(Y_i, start_month=start_month_i,
                                                      relevant_X_lags=X_lags[:bool_use_AR_lags], horizon=horizon)
                        sdcm_noexog_alt.params = sdcm_noexog_alt._params_from_vector(param_ests_noexog[:,i - 1])
                        sdcm_noexog_alt.estimate(print_results=verbose, construct_std_errors=False)

                        if sdcm_noexog_alt.llh > sdcm_noexog.llh or np.isnan(sdcm_noexog.llh):
                            sdcm_noexog = sdcm_noexog_alt
                    if use_random_search:
                        sdcm_noexog_rs = _make_model(Y_i, start_month=start_month_i,
                                                     relevant_X_lags=X_lags[:bool_use_AR_lags], horizon=horizon)
                        sdcm_noexog_rs.random_start_search(construct_std_errors=False, print_progress=verbose,
                                                           n_draws=n_draws, n_best=n_best)

                        if sdcm_noexog_rs.llh > sdcm_noexog.llh or np.isnan(sdcm_noexog.llh):
                            sdcm_noexog = sdcm_noexog_rs
                    param_ests_noexog[:,i]= sdcm_noexog.get_param_vector()
            if include_X:
                sdcm_fin = _make_model(Y_i, X=X_i, start_month=start_month_i,
                                       relevant_X_lags=X_lags, X_labels=X_labels, horizon=horizon, num_PCA=num_PCA)
                if method == 'G2S':
                    if num_comps > 0:
                        flat_g2s = sdcm_fin._params_to_vector(sdcm_fin.params)
                        flat_g2s[:param_ests_noexog[:,i].shape[0]] = param_ests_noexog[:,i]
                        sdcm_fin.params = sdcm_fin._params_from_vector(flat_g2s)
                    sdcm_fin.general_to_specific_grouped(sig_level=sig_level, bool_pre_estimate=False,
                                                         bool_random_search=False, print_progress=verbose)
                elif method == 'Lasso' and num_comps == 0:
                    sdcm_fin.estimate_lasso_gridsearch(lambda_grid, print_results=verbose, print_subresults=verbose)
                param_ests_vec = sdcm_fin.get_param_vector()
            else:
                sdcm_fin = sdcm_noexog
                param_ests_vec = param_ests_noexog[:,i]

        forecast = sdcm_fin.construct_forecast()
        forecast_error= Y_vec_hor[i + insample_length + horizon - 1] - forecast
        #print(i + 1, end=",")
        #end = timeit.default_timer()
        #print(f"computation time: {end - start:.2f}s")
        return i, param_ests_vec, forecast, forecast_error

    if bool_parallelize:
        results = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
                delayed(run_single_estimation_and_forecast)(i)
                for i in range(outsample_length_hor)
        )
    else:
        results = [
            run_single_estimation_and_forecast(i)
            for i in range(outsample_length_hor)
        ]

    for j, param_vec, forecast, forecast_error in results:
        forecast_vec[j] = forecast
        forecast_error_vec[j] = forecast_error
        if not bool_use_pre_estim:
            param_ests[:,j] = param_vec

    mse = np.mean(np.square(forecast_error_vec))

    ## Save
    str_ar_lags = '' if (ar_lags == 3 or ar_lags == 0) else str(ar_lags)
    string_info = (method * include_X + str(num_comps) + 'componentsSetting' + setting + bool_use_AR_lags * (
                '_AR' + str_ar_lags)
                   + bool_use_seas_dummies * '_seasdummies' + bool_use_anomalies * '_anomalies' + 'withExog' * include_X
                   + bool_use_PCA * 'PCA')
    string_info_hor = string_info + 'horizon' + str(horizon)

    if save_results:
        save_forecast_results(output_folder,sdcm_dummy.tot_labels, string_dates_hor[insample_length + horizon - 1:], param_ests,
                              forecast_vec, forecast_error_vec, string_info_hor, bool_use_pre_estim)
    print(string_info_hor, np.round(mse,3))

    return mse,string_info,forecast_vec

