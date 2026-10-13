#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
@author: Janneke van Brummelen
"""

###########################################################
### Imports  
import numpy as np
import scipy.optimize as opt 
import scipy.stats as st   
import numba as nb
import math 
import timeit 
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import statsmodels.api as sm 
from dataclasses import dataclass 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

EPS = 1e-8

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softplus_inv(y):
    return np.log(np.expm1(y))

def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x) - np.log1p(-x)

@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(
    nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64,nb.float64[:,:],
    nb.float64[:,:], nb.float64, nb.int64, nb.int64, nb.int64, nb.int64, nb.int64), cache=True, fastmath=True)
def _filter_loop(y, X, beta, omega, mat_phi, mat_theta, start, T, num_comps,
                 max_ar_lag, max_ma_lag, max_lag):

    paths = np.zeros((T+1, num_comps))
    residuals = np.zeros(T) 
    paths[0,0] = start

    for i in range(1,T+1):
        # prediction error
        mu = 0.0
        for j in range(num_comps):
            mu += paths[i-1, j]

        xb = 0.0
        for j in range(X.shape[1]):
            xb += beta[j] * X[i-1, j]

        residuals[i-1] = y[i-1] - mu - xb - omega

        if i >= max_lag:
            for j in range(num_comps):
                val = 0.0
                # AR part
                for k in range(max_ar_lag):
                    val += paths[i-max_ar_lag+k, j] * mat_phi[k, j]
                # MA part
                for k in range(max_ma_lag):
                    val += mat_theta[k, j] * residuals[i-max_ma_lag+k]
                paths[i, j] = val
        else:
            r = min(i, max_ar_lag)
            q = min(i, max_ma_lag) 
            for j in range(num_comps):
                val = 0.0
                for k in range(r):
                    val += paths[i-r+k,j] * mat_phi[max_ar_lag-r+k,j]
                for k in range(q):
                    val += mat_theta[max_ma_lag-q+k,j] * residuals[i-q+k]
                paths[i,j] = val

    return residuals, paths

###########################################################
### vh= _gh_stepsize(vP)
def _gh_stepsize(vP):
    """
    Purpose:
        Calculate stepsize close (but not too close) to machine precision

    Inputs:
        vP      1D array of parameters

    Return value:
        vh      1D array of step sizes
    """
    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
    vh= np.maximum(vh, 5e-6)       # Don't go too small

    return vh

###########################################################
### mG= jacobian_2sided(fun, vP, *args)
def jacobian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical jacobian, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, return 1D array of size iN
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mG      iN x iP matrix with jacobian

    See also:
      numdifftools.Jacobian(), for similar output
    """
    iP = vP.shape[0]
    vP= vP.reshape(iP)      # Ensure vP is 1D-array

    vF = fun(vP, *args)     # evaluate function, only to get size
    iN= vF.size

    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    mGp = np.zeros((iN, iP))
    mGm = np.zeros((iN, iP))

    for i in range(iP):     # Find f(x+h), f(x-h)
        mGp[:,i] = fun(vP+mh[i], *args)
        mGm[:,i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    mG= (mGp - mGm) / (vhr + vhl)  # Get central jacobian

    return mG
 
###########################################################
### mH= hessian_2sided(fun, vP, *args)
def hessian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical hessian, using a 2-sided numerical difference

    Author:
      Kevin Sheppard, adapted by Charles Bos

    Source:
      https://www.kevinsheppard.com/Python_for_Econometrics

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mH      iP x iP matrix with symmetric hessian
    """
    iP = vP.shape[0]
    vP= vP.reshape(iP)    # Ensure vP is 1D-array

    f = fun(vP, *args)
    vh= _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j,i] = fmm[i,j]

    vh = vh.reshape((iP,1))
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    return mH

@dataclass
class ModelParameters:
    sigma:  float
    omega:  float
    phi:    list[np.ndarray]            # phi[k] shape (vec_autoreg_lags[k],)
    theta:  list[np.ndarray]            # theta[k] shape (vec_ma_lags[k],)
    beta:   np.ndarray                  # shape (num_X_cols,)

    def copy(self):
        return ModelParameters(
            sigma=self.sigma,
            omega=self.omega,
            beta=self.beta.copy(),
            phi=[p.copy() for p in self.phi],
            theta=[t.copy() for t in self.theta]
        )

@dataclass
class FixedParameters:
    sigma:  bool
    omega:  bool
    phi:    list[np.ndarray]   # phi[k] shape (vec_autoreg_lags[k],)
    theta:  list[np.ndarray]   # theta[k] shape (vec_ma_lags[k],)
    beta:   np.ndarray    # shape (num_X_cols,)
    
    def copy(self):
        return FixedParameters(
            sigma=self.sigma,
            omega=self.omega,
            beta=self.beta.copy(),
            phi=[p.copy() for p in self.phi],
            theta=[t.copy() for t in self.theta]
        )
      
class ScoreDrivenComponentModel:
    """
    Score-driven component model with regressors
    
    Model:
        y_t = mu_t + x_t' beta + eps_t
        mu_t = alpha_t + gamma_t + psi_t
        alpha_{t+1} = omega + alpha_t + theta_alpha s_t
        gamma_t = gamma_t^{(1)} + ... + gamma_t^{(s)}
        psi_t = psi_t^{(1)} + ... + psi_t^{(r)}
        gamma_{t+1}^{(i)} = phi_{gamma,1}^{(i)} gamma_{t}^{(i)} 
             + phi_{gamma,2}^{(i)} gamma_{t-1}^{(i)} + theta_gamma^{(i)} s_t
        psi_{t+1}^{(i)} = phi_{psi,1}^{(i)} psi_{t}^{(i)} 
             + phi_{psi,2}^{(i)} psi_{t-1}^{(i)} + theta_psi^{(i)} s_t
    Where:
        s_t = score function
        x_t = vector of explanatory variables
        More than two autoregressive lags and one MA lag can be used
                              
    """
    def __init__(self, y, X=None, dates=None, T_insample=None, num_comps=0, vec_autoreg_lags=None,
                 vec_ma_lags=None, init_params=None, horizon=1, burn=0, num_avg=0, density='Gaussian', 
                 cycle_per_restrictions=None, param_restrictions=None, relevant_X_lags=None,
                 X_labels=None, max_lag=36, bool_include_AR=False, bool_seas_dummies=False,
                 num_PCA=None, search_bounds=None, start_month=0):
        """
        y : 1-d array of observations, shape (T,)
        
        X : None or 2-d array of explanatory variables (may not include y), shape (T,N) 
            
        dates : vector of dates in matplotlib dates format to use for plotting, if None use indices by default
        
        T_insample : number of observations to use for in-sample period (this includes 
                     the time observations 'lost' due to lags)
        
        num_comps : number of components to use 
        
        vec_autoreg_lags : 1-d array with 'num_comps' elements, each element contains
                           autoregressive lag order to use for each component
        
        vec_ma_lags : 1-d array with 'num_comps' elements, each element contains
                           ma lag order to use for each component
        
        init_params : dict or None
            Keys: sigma, omega, phi_k, theta_k, beta, 'beta_seas' for only seasonal dummies
         
        horizon : which forecast horizon will be used, only relevant if regressors
              are included in the model, because only 
            
        burn : number of observations to burn, integer
        
        num_avg : number of observations to use for starting value of first component
        
        density : which density to use
            Options: 'Gaussian' 
            
        cycle_per_restrictions : dictionary with cycle period restrictions,
                       use index of components as keys, and have value (lower bound, upper bound)
                       for bounds on cycle period, or None to only restrict component
                       to be cyclical without restricting periodicity. 
                
        param_restrictions : dictionary with restrictions to impose, use param 
                            keys and use 'fix_all' for any parameter, 'fix_frequency' for
                            phi parameters with 2 lags
                            or use list of indices to fix certain parameters
        
        relevant_X_lags : list with N elements. Each element is itself a list of 
                          the relevant lags for each explanatory variable. Note:
                          if bool_include_AR is True, first element should concern
                          which AR lags to include. 
                          
        X_labels : list with the names of the explanatory variables
        
        max_lag : integer, maximum number of lags to include
        
        bool_include_AR : boolean, True if y should be included as explanatory variable
        
        bool_seas_dummies : boolean, True if seasonal dummies should be included
         
        num_PCA : integer, number of PCA components to include. If strictly positive, 
                           PCA is used to summarize information present in explanatory
                           variables at each lag order with 'num_PCA' PC's
        
        search_bounds : dictionary or None
            Overrides default search bounds used in random initialisation.
            Structure:
                'phi_k'    : dict with lag index j as key and tuple (lower, upper) as value,
                             where k is the 1-indexed component number
                             default: {'phi_1': {0: (0.9, 1.9), 1: (-1.0, -0.004)}}
                'theta_k'  : dict with lag index j as key and tuple (lower, upper) as value,
                             where k is the 1-indexed component number
                             default: {'theta_2': {0: (0.001, 1.5)}}
                'sigma'    : tuple (lower, upper), default: (0.06,0.11)
                'beta'     : tuple (lower, upper) for all selectable regressors, default: (-0.2,0.2)
                'beta_seas': tuple (lower, upper) for seasonal dummy parameters, default: None
            User-provided values override defaults for matching keys. Default bounds 
            are used for any key not provided. Pass None for a key to remove it from 
            the search bounds entirely.
            
        start_month : index (0, January to 11, December) of which month is the first month of the sample
                        used for constructing seasonal dummies, such that first dummy
                        corresponds to January
        """
         
        self.y = np.asarray(y)  
        self.T = self.y.shape[0]
        if X is None:
            X = np.zeros((self.T,0))
        else:
            assert X.shape[0] == y.shape[0]
        if T_insample is None:
            self.T_insample = self.T
        else:
            self.T_insample = T_insample
            assert T_insample <= self.T 
        self.bool_include_AR = bool_include_AR 
        self.bool_seas_dummies = bool_seas_dummies
        self.dates = dates
          
        self.N = X.shape[1] + int(bool_include_AR)
        
        self.num_comps = num_comps
        self.vec_autoreg_lags = vec_autoreg_lags if num_comps > 0 else np.array([], dtype=int)
        self.vec_ma_lags = vec_ma_lags if num_comps > 0 else np.array([], dtype=int)
    
        self.burn = burn
        self.horizon = horizon
        self.max_lag = max_lag
        self.T_insample_fin = self.T_insample - self.max_lag
        self.relevant_X_lags = [
            [j for j in lags if (i == 0 and self.bool_include_AR) or j >= self.horizon]
            for i, lags in enumerate(relevant_X_lags)] if relevant_X_lags is not None else None
        self.num_PCA = num_PCA 
          
        self.X_labels = X_labels if X_labels is not None else [] 
        self.param_restrictions = param_restrictions.copy() if param_restrictions is not None else {}
        self.cycle_per_restrictions = cycle_per_restrictions if cycle_per_restrictions is not None else {}
        self._check_inputs(X)
        
        X_trans = self._transform_X(X)
        self.X_full, self.X = self._build_X_lags(X_trans,start_month)  
        self.y = self.y[self.max_lag:]  
        self.y_full = self.y.copy()
        self.y = self.y[:self.T_insample_fin] 
        self.dates_insample = self.dates[self.max_lag:self.T_insample] if self.dates is not None else None
 
        self.N_tot = self.X.shape[1]
        self.params = self._initialize_params(init_params) 
        self._check_params(self.params)  
        
        self.fixed = self._get_fix_indic() 
        self.tot_labels = self._construct_label_vec() 
          
        self.num_avg = num_avg 
        self.bool_trend = (self.num_comps > 0 and
                   self.vec_autoreg_lags[0] == 1 and 
                   'phi_1' in self.param_restrictions and 
                   self.param_restrictions['phi_1'] == 'fix_all' and
                   self.params.phi[0][0] == 1.0 and
                   ('theta_1' not in self.param_restrictions or 
                    self.param_restrictions['theta_1'] != 'fix_all' or
                    self.params.theta[0][0] != 0.0))
        self.start = np.mean(self.y[:self.num_avg]) if self.bool_trend else 0.0
        self.density = density 
        
        self.paths = np.zeros((self.T_insample_fin, self.num_comps)) 
        self.residuals = None
        self.llh = None
        self.success = None
        self.std_errors = None  
        
        default_search_bounds = {
            'sigma':  (0.06,0.11),
            **{f'phi_{k+1}': {j: (0.9, 1.9) if j == 0 else (-1.0, -0.004) 
                              for j in range(self.vec_autoreg_lags[k])}
               for k in range(self.num_comps)},
            **{f'theta_{k+1}': {j: (0.001, 1.5) 
                                for j in range(self.vec_ma_lags[k])}
               for k in range(self.num_comps)},
            'beta': (-0.2, 0.2),
        }
        merged = {**default_search_bounds, **(search_bounds or {})}
        self.search_bounds = {k: v for k, v in merged.items() if v is not None}
          
    def _transform_X(self, X, significance_level=0.05, standardize=True):
        
        crit_value = st.norm.ppf(1-significance_level/2)
        time_trend = np.arange(self.T).reshape(-1,1)
        X_detrended = X.copy()
        mat_cons_and_trend = sm.add_constant(time_trend.reshape(-1,1))
         
        # self.N-int(self.bool_include_AR) because self.N includes the AR lag
        # if it is present, but AR lags should not be transformed
        for i in range(self.N-int(self.bool_include_AR)):  
            model = sm.regression.linear_model.OLS(X[:self.T_insample,i],mat_cons_and_trend[:self.T_insample,:])
            results = model.fit()
            slope = results.params[1]
            if np.abs(slope)/results.bse[1] > crit_value:
                intercept = results.params[0]
                X_detrended[:,i] = X[:,i] - intercept - slope * time_trend.reshape(-1,) 
        
        if standardize:
            X_detrended = (X_detrended- np.mean(X_detrended[:self.T_insample,:], axis=0))/np.std(X_detrended[:self.T_insample,:], axis=0)
    
        return X_detrended
    
    def _build_AR_lags(self):
        return np.column_stack([self.y[self.max_lag-j:-j] for j in self.relevant_X_lags[0]]) 
    
    def _build_X_lags(self, X, start_month):
          
        if not self.num_PCA: 
            X_raw = np.hstack((self.y.reshape(-1,1),X)) if self.bool_include_AR else X
            
            cols = [X_raw[self.max_lag-j:-j, i] for i in range(X_raw.shape[1]) 
                for j in self.relevant_X_lags[i]] 
            X_mat = np.column_stack(cols) if X_raw.shape[1]>0 else np.zeros((self.T-self.max_lag,0))
            
        else: 
            X_mat_PCA = self._build_PCA_lags(X)
            X_mat = np.hstack((self._build_AR_lags(), X_mat_PCA)) if self.bool_include_AR else X_mat_PCA
            min_lag, max_lag = self._get_X_lag_range()
            pc_lags = list(range(min_lag, max_lag + 1))
            if self.bool_include_AR:
                self.relevant_X_lags = [self.relevant_X_lags[0]] + [pc_lags] * self.num_PCA
            else:
                self.relevant_X_lags = [pc_lags] * self.num_PCA
            self.N = len(self.relevant_X_lags)
            self.X_labels = [f'PC{i+1}' for i in range(self.num_PCA)]
            
        if self.bool_seas_dummies:
            dum_mat = np.roll(np.eye(12)[:, :-1],start_month)
            seas_dum = np.tile(dum_mat, (int(np.ceil((self.T - self.max_lag) / 12)), 1)) 
            seas_dum = seas_dum[:self.T-self.max_lag,:]
            X_mat = np.hstack((seas_dum,X_mat))
         
        return X_mat, X_mat[:self.T_insample_fin,:]
    
    def _get_X_lag_range(self):
        vecs = self.relevant_X_lags[int(self.bool_include_AR):]
        min_lag = int(min(np.min(v) for v in vecs if len(v) > 0))
        max_lag = int(max(np.max(v) for v in vecs if len(v) > 0))
        return min_lag, max_lag
        
    def _build_PCA_lags(self, X):
        min_PCA_lag, max_PCA_lag = self._get_X_lag_range() 
        
        # Store transformed results per lag: pca_results[j] has shape (T', num_PCA)
        pca_results = []
        for lag in range(min_PCA_lag, max_PCA_lag + 1): 
            vars_at_lag = [i-int(self.bool_include_AR) for i in range(int(self.bool_include_AR), self.N)
                           if lag in self.relevant_X_lags[i]] 
            if len(vars_at_lag) == 0:
                continue
            
            X_lag_mat = np.column_stack([X[self.max_lag - lag:-lag, i] for i in vars_at_lag])
            
            pca = PCA(n_components=self.num_PCA, svd_solver='full')
            pca.fit(X_lag_mat[:self.T_insample_fin])
            pca_results.append(pca.transform(X_lag_mat))
             
        # Reorder: all lags of PC1, then all lags of PC2, etc. 
        return np.hstack([np.column_stack([r[:, i] for r in pca_results]) 
                      for i in range(self.num_PCA)])
    
    def _initialize_params(self, init_params):
        phi = []  
        for k, p in enumerate(self.vec_autoreg_lags):
            if k == 0 and p == 1:
                phi.append(np.array([1.0]))
            elif p == 2 and k+1 in self.cycle_per_restrictions:
                phi2 = -0.35  # default phi2
                if self.cycle_per_restrictions[k+1] is not None:
                    bounds = self.cycle_per_restrictions[k+1]
                    target_period = np.mean(bounds) 
                    phi1 = 2 * np.sqrt(-phi2) * np.cos(2 * np.pi / target_period)
                else:
                    phi1 = 2 * np.sqrt(-phi2) * np.cos(2 * np.pi / 14) 
                phi.append(np.array([phi1, phi2]))
            else:
                phi.append(np.array([(0.9)**i*(-1)**(i+1) for i in range(1, p + 1)]))
    
        start_params = ModelParameters(
                sigma=0.1,
                omega=0.0,
                beta=np.zeros(self.X.shape[1]),
                phi=phi,
                theta=[np.ones(q)*0.1 for q in self.vec_ma_lags]
            ) 
        
        if init_params is not None:
            for key, value in init_params.items():
                if key == 'sigma':
                    start_params.sigma = value
                elif key == 'omega':
                    start_params.omega = value 
                elif key == 'beta_seas' and self.bool_seas_dummies:
                    start_params.beta[:11] = value  
                elif key == 'beta':
                    start_params.beta[11*int(self.bool_seas_dummies):] = value
                elif key.startswith('phi_'):
                    k = int(key.split('_')[1]) - 1
                    start_params.phi[k] = value
                elif key.startswith('theta_'):
                    k = int(key.split('_')[1]) - 1
                    start_params.theta[k] = value
        # override phi_1 for frequency-restricted components to match target period
        if self.cycle_per_restrictions is not None:
            for k_key, bounds in self.cycle_per_restrictions.items():
                k = k_key - 1  # convert to 0-indexed
                if self.vec_autoreg_lags[k] == 2 and bounds is not None:
                    target_period = np.mean(bounds)
                    phi2 = start_params.phi[k][1]
                    start_params.phi[k][0] = 2 * np.sqrt(-phi2) * np.cos(2 * np.pi / target_period)
        
        return start_params
 
    def _check_params(self, params): 
        if params.beta.shape != (self.N_tot,):
            raise ValueError(f"beta has wrong shape: expected ({self.X.shape[1]},), got {params.beta.shape}")
        if not isinstance(params.sigma, float):
            raise ValueError("sigma must be a scalar")
        if not isinstance(params.omega, float):
            raise ValueError("omega must be a scalar")
        if len(params.phi) != self.num_comps:
            raise ValueError(f"phi must have {self.num_comps} components, got {len(params.phi)}")
        if len(params.theta) != self.num_comps:
            raise ValueError(f"theta must have {self.num_comps} components, got {len(params.theta)}")
        for k in range(self.num_comps):
            if params.phi[k].shape != (self.vec_autoreg_lags[k],):
                raise ValueError(f"phi[{k}] has wrong shape: expected ({self.vec_autoreg_lags[k]},), got {params.phi[k].shape}")
            if params.theta[k].shape != (self.vec_ma_lags[k],):
                raise ValueError(f"theta[{k}] has wrong shape: expected ({self.vec_ma_lags[k]},), got {params.theta[k].shape}")
        
    def _check_inputs(self, X): 
        if self.T <= self.burn:
            raise ValueError("the number of burned observations is larger than the total number of observations")
        if not np.isfinite(self.y).all():
            raise ValueError("Data contains NaN or Inf values.")
        if not all(((v in ['fix_all','fix_frequency']) or isinstance(v, list)) for v in self.param_restrictions.values()):
            raise ValueError("Restrictions contain forbidden types.")
        valid_keys = ['sigma', 'omega', 'beta', 'beta_seas'] + \
             [f'phi_{k+1}' for k in range(self.num_comps)] + \
             [f'theta_{k+1}' for k in range(self.num_comps)]
        if not all(k in valid_keys for k in self.param_restrictions.keys()):
            raise ValueError("Restrictions contain forbidden keys.")  
        if X.shape[1] > 0 and self.relevant_X_lags is None:
            raise ValueError("relevant_X_lags is missing from inputs.") 
        if self.relevant_X_lags is not None and len(self.relevant_X_lags)-int(self.bool_include_AR) != X.shape[1]:
            raise ValueError("Mismatch between X columns and relevant_X_lags")
        if self.relevant_X_lags is not None:
            for i, lags in enumerate(self.relevant_X_lags):
                if np.isscalar(lags):
                    raise ValueError(f"relevant_X_lags[{i}] should be an array of lags, not a scalar")
        for key, value in self.param_restrictions.items():
            if value == 'fix_frequency':
                k = int(key.split('_')[1])
                if k not in self.cycle_per_restrictions or self.cycle_per_restrictions[k] is None:
                    raise ValueError(f"fix_frequency for {key} requires non-None bounds in cycle_per_restrictions.")
            
    def _construct_main_labels(self): 
        main_labels = ['sigma', 'omega','phi','theta'] 

        if self.bool_include_AR:
            main_labels += ['y']
        if not self.num_PCA:
            main_labels += self.X_labels
        else:
            main_labels += ['PC '+str(i+1) for i in range(self.num_PCA)]

        return main_labels
    
    def _construct_label_vec(self):
        labels = []
        
        # sigma and omega
        labels.append(np.array(['sigma^2', 'omega']))
         
        # phi: phi_k_lag
        for k in range(self.num_comps):
            for lag in range(1, self.vec_autoreg_lags[k] + 1):
                labels.append(np.array([f'phi_{k+1}_{lag}']))
        
        # theta: theta_k_lag
        for k in range(self.num_comps):
            for lag in range(1, self.vec_ma_lags[k] + 1):
                labels.append(np.array([f'theta_{k+1}_{lag}']))
         
        # beta: seasonal dummies first, then variable names with lag suffixes
        beta_labels = []
    
        if self.bool_seas_dummies:
            beta_labels += [f'dummy {i+1}' for i in range(11)]
        
        if self.N > 0:
            var_names = []
            if self.bool_include_AR:
                var_names.append('y') 
            var_names += self.X_labels 
            beta_labels += [f'{var}_{lag}' for var, lags in zip(var_names, self.relevant_X_lags)
                            for lag in lags]
             
        labels.append(np.array(beta_labels)) 
        flat_labels = np.concatenate(labels).astype(object)
        
        # add '(fixed)' to fixed parameters
        flat_mask = self._build_flat_mask()  
        flat_labels = np.where(flat_mask, flat_labels + ' (fixed)', flat_labels)
        
        return flat_labels

    def _get_fix_indic(self):
        """Construct FixedParameters based on param_restrictions and update
            starting values of fixed parameters where necessary."""
        
        fixed = FixedParameters(
            sigma=False,
            omega=False,
            beta=np.zeros(self.X.shape[1], dtype=bool),
            phi=[np.zeros(p, dtype=bool) for p in self.vec_autoreg_lags],
            theta=[np.zeros(q, dtype=bool) for q in self.vec_ma_lags]
        )    
        
        if self.param_restrictions is None:
            return fixed

        for key, value in self.param_restrictions.items(): 
             if key == 'sigma':
                fixed.sigma = True
             elif key == 'omega':
                fixed.omega = True
             elif key == 'beta':
                if value == 'fix_all':
                    fixed.beta[:] = True
                elif isinstance(value, list):
                    fixed.beta[value] = True
             elif key.startswith('phi_'):
                k = int(key.split('_')[1]) - 1
                if value == 'fix_all':
                    fixed.phi[k][:] = True
                elif value == 'fix_frequency':
                    # phi_1 is deterministic function of phi_2 and frequency
                    # so only fix phi_1, phi_2 remains free
                    fixed.phi[k][0] = True
                elif isinstance(value, list):
                    fixed.phi[k][value] = True
             elif key.startswith('theta_'):
                k = int(key.split('_')[1]) - 1
                if value == 'fix_all':
                    fixed.theta[k][:] = True
                elif isinstance(value, list):
                    fixed.theta[k][value] = True

        return fixed
    
    def _build_flat_mask(self, fixed=None):
        """Return a boolean mask for the flattened parameter vector."""
        if fixed is None:
            fixed = self.fixed
        
        return np.concatenate([np.array([fixed.sigma, fixed.omega]), *fixed.phi, 
                               *fixed.theta, fixed.beta])
        
    def _extract_free_params(self, flat_params, fixed=None):
        """Return vector of parameters that are not fixed.""" 
        if fixed is None:
            fixed = self.fixed
        
        return flat_params[~self._build_flat_mask(fixed=fixed)]
    
    def _get_no_free_params(self,fixed=None):
        if fixed is None:
            fixed = self.fixed
        return np.sum(~self._build_flat_mask(fixed))
    
    def _insert_fixed_params(self, free_params, full_params, fixed=None):
        """Rebuild full flattened parameter vector including fixed entries.""" 
        if fixed is None:
            fixed = self.fixed
    
        full_vec = self._params_to_vector(full_params) 
        full_vec[~self._build_flat_mask(fixed=fixed)] = free_params
         
        return full_vec 
    
    def _insert_nans_fixed(self, free_vals, fixed=None):
        """Construct vector with nans corresponding to fixed entries."""
        if fixed is None:
            fixed = self.fixed
        
        nan_params = ModelParameters(
            sigma=np.nan,
            omega=np.nan,
            beta=np.full(self.N_tot, np.nan),
            phi=[np.full(p, np.nan) for p in self.vec_autoreg_lags],
            theta=[np.full(q, np.nan) for q in self.vec_ma_lags]
        )
        
        full_vec = self._params_to_vector(nan_params)
        mask = self._build_flat_mask(fixed=fixed)
        full_vec[~mask] = free_vals
        
        return full_vec
    
    def _params_to_vector(self, params):
        return np.concatenate(
            [np.array([params.sigma, params.omega]),
             *params.phi,
             *params.theta,
             params.beta])
    
    def _params_from_vector(self, vec):
        idx = 0
        sigma = vec[idx]; idx += 1
        omega = vec[idx]; idx += 1
        phi   = []
        for p in self.vec_autoreg_lags:
            phi.append(vec[idx:idx + p]); idx += p
        theta = []
        for q in self.vec_ma_lags:
            theta.append(vec[idx:idx + q]); idx += q
        beta  = vec[idx:idx + self.N_tot]
        return ModelParameters(sigma=sigma, omega=omega, beta=beta, phi=phi, theta=theta)
      
    def get_param_vector(self):
        """Return the current parameter vector""" 
        return self._params_to_vector(self.params)
        
    def _calc_std_errors(self, params=None, fixed=None, robust=True, check_hess_posdef = True):
        """Calculate standard errors, returning vector with nans for fixed parameters"""
        if fixed is None:
            fixed = self.fixed
        if params is None:
            params = self.params
        
        T_effective = self.T_insample_fin - self.burn

        free_params = self._extract_free_params(self._params_to_vector(params),fixed=fixed)
        calc_llh_free = lambda free_params: self.calc_loglikelihood(self._params_from_vector(self._insert_fixed_params(free_params,params,fixed=fixed)))
        calc_avg_llh_free = lambda free_params: np.mean(calc_llh_free(free_params))
        hessian = hessian_2sided(calc_avg_llh_free, free_params)

        if check_hess_posdef:
            try:
                np.linalg.cholesky(-hessian)
            except np.linalg.LinAlgError:
                return np.full(len(self._params_to_vector(params)), np.nan)
        hessian_inv = np.linalg.inv(hessian)

        if robust:
            jacobian = jacobian_2sided(calc_llh_free, free_params)
            jacobian_inner = jacobian.T @ jacobian / T_effective
            cov = hessian_inv @ jacobian_inner @ hessian_inv/T_effective
        else: 
            cov = -hessian_inv/T_effective  
             
        std_error_free = np.sqrt(np.diagonal(cov)).reshape(-1,)
         
        return self._insert_nans_fixed(std_error_free,fixed=fixed)

    def _transform_params(self, params):
        """Transform parameters in dict to ensure the parameters stay in certain regions"""
        params_trans = params.copy()
        
        # ---- sigma ----
        params_trans.sigma = softplus_inv(params_trans.sigma - EPS)
          
        # ---- phi ----
        if self.num_comps > 0:   
            for i in range(self.num_comps):
                if self.vec_autoreg_lags[i] == 2 and i+1 in self.cycle_per_restrictions:
                    phi1 = min(params_trans.phi[i][1], -EPS)  # ensure strictly negative
                    den = np.sqrt(-phi1)
                    term = params_trans.phi[i][0] / (2 * den)
                    # set phi_1 equal to periodicity
                    term = np.clip(term, -1.0+EPS, 1.0-EPS)
                    params_trans.phi[i][0] = 2 * np.pi / math.acos(term)
                    x = np.clip(-params_trans.phi[i][1], EPS, 1 - EPS)
                    params_trans.phi[i][1] = logit(x)

                    if self.cycle_per_restrictions[i+1] is not None:
                        lb = self.cycle_per_restrictions[i+1][0]
                        ub_min_lb = self.cycle_per_restrictions[i+1][1] - lb 
                        help_term = (params_trans.phi[i][0] - lb)/ub_min_lb if ub_min_lb > 0 else 1
                        x = np.clip(help_term, EPS, 1 - EPS)
                        params_trans.phi[i][0] = logit(help_term)
                    else:
                        params_trans.phi[i][0] = softplus_inv(params_trans.phi[i][0])
                            
        # ---- theta ----
            for i in range(self.num_comps):
                if not np.any(self.fixed.theta[i]):
                    params_trans.theta[i] = softplus_inv(params_trans.theta[i])
                    
        return params_trans

    def _transform_params_back(self, params_trans):
        """Transform parameters back in dict to ensure the parameters stay in certain regions""" 
        params = params_trans.copy()
        
        # ---- sigma ----
        params.sigma = EPS + softplus(params.sigma)
        
        # ---- phi ----
        if self.num_comps > 0:   
            for i in range(self.num_comps):
                if self.vec_autoreg_lags[i] == 2 and i+1 in self.cycle_per_restrictions: 
                    if self.cycle_per_restrictions[i+1] is not None:     
                        lb = self.cycle_per_restrictions[i+1][0]
                        ub_min_lb = self.cycle_per_restrictions[i+1][1] - lb
                        x = np.clip(params.phi[i][0], -50, 50)
                        params.phi[i][0] = lb + ub_min_lb*expit(x)
                    else:
                        params.phi[i][0] = softplus(params.phi[i][0])
                    x = np.clip(params.phi[i][1], -50, 50)
                    params.phi[i][1] = - expit(x)
                    val = np.minimum(params.phi[i][1], -1e-12)
                    den = np.clip(params.phi[i][0], 1e-8, None)
                    params.phi[i][0] = 2 * np.sqrt(-val) * np.cos(2 * np.pi/ den)
 
        # ---- theta ----
            for i in range(self.num_comps):
                if not np.any(self.fixed.theta[i]):
                    params.theta[i] = softplus(params.theta[i])
        
        return params  
     
    def construct_filtered_seqs(self, params):
        if self.num_comps == 0:
            residuals = self.y - self.X @ params.beta - params.omega
            return residuals, np.zeros((self.T_insample_fin + 1, 0))
    
        max_ar_lag = np.max(self.vec_autoreg_lags)
        max_ma_lag = np.max(self.vec_ma_lags)
        max_lag = max(max_ar_lag, max_ma_lag)
    
        mat_phi = np.zeros((max_ar_lag, self.num_comps))
        mat_theta = np.zeros((max_ma_lag, self.num_comps)) 
        for i in range(self.num_comps):
            mat_phi[-self.vec_autoreg_lags[i]:, i] = np.flip(params.phi[i])
            mat_theta[-self.vec_ma_lags[i]:, i] = np.flip(params.theta[i])
    
        residuals, paths = _filter_loop(self.y, self.X, params.beta, params.omega,
            mat_phi, mat_theta, self.start, self.T_insample_fin, self.num_comps,
            max_ar_lag, max_ma_lag, max_lag)
    
        return residuals, paths

    def calc_loglikelihood(self, params):
        """Calculate for given parameters"""
            
        resid = self.construct_filtered_seqs(params)[0] 
        LL_vec = np.square(resid).reshape(-1,)  
         
        LL_vec = -0.5*np.log(2*np.pi) - 0.5 * np.log(params.sigma) - 0.5* LL_vec/params.sigma
        
        return LL_vec[self.burn:] 
    
    def calc_loglikelihood_trans(self, params_trans_flattened):
        """Calculate for given transformed parameters"""
        params_trans = self._params_from_vector(params_trans_flattened)
        params = self._transform_params_back(params_trans)
        
        return self.calc_loglikelihood(params)
      
    def _estimate_internal(self, init_params, fixed):
        """Estimate model parameters using QML"""  
        
        params_trans = self._transform_params(init_params) 
        init_params_trans_flatten_free = self._extract_free_params(self._params_to_vector(params_trans),fixed=fixed) 
        
        avg_neg_llh = lambda theta: -np.mean(self.calc_loglikelihood_trans(self._insert_fixed_params(theta,params_trans,fixed=fixed)))
        res = opt.minimize(avg_neg_llh, init_params_trans_flatten_free, method="BFGS")
        
        params = self._transform_params_back(self._params_from_vector(self._insert_fixed_params(res.x,params_trans,fixed=fixed)))
        llh = np.sum(self.calc_loglikelihood(params))
        residuals, paths = self.construct_filtered_seqs(params)
        paths = paths[:-1,:] # remove final value, as it is prediction
        success = res.success

        return params, llh, residuals[self.burn:], paths, success, res.message
        
    def estimate(self, construct_std_errors = True, print_results = True, check_hess_posdef = True):
        """Estimate model parameters using ML"""  
        
        self.params, self.llh, self.residuals, self.paths, self.success, message = self._estimate_internal(self.params,self.fixed)
        if construct_std_errors:
            self.std_errors = self._calc_std_errors(robust=False,check_hess_posdef=check_hess_posdef)
        if print_results:
            if not self.success:
                print(message)
            self.summarize_results(title='Estimation results estimation')  
    
    def estimate_lasso_gridsearch(self, lambda_grid, criterion="BIC", print_results=True, print_subresults=True):
        """LASSO grid search for num_comps = 0 case"""
    
        if self.num_comps != 0:
            raise ValueError("LASSO only implemented for num_comps = 0") 
            
        y = self.y[self.burn:]
        X = self.X[self.burn:, :]
    
        best_ic, best_lambda, best_params, best_resid, best_k = np.inf, None, None, None, None
        T_eff = len(y)
    
        if criterion == "AIC":
            ic_func = lambda llh, k: -2 * llh + 2 * k
        elif criterion == "BIC":
            ic_func = lambda llh, k: -2 * llh + np.log(T_eff) * k
        else:
            raise ValueError("criterion must be AIC or BIC")
            
        if not print_results:
            print_subresults = False
    
        for lam in sorted(lambda_grid): 
            clf = Lasso(alpha=lam, fit_intercept=True, max_iter=10000)
            clf.fit(X, y)
    
            beta_hat = clf.coef_
            omega_hat = clf.intercept_ 
            resid = y - (X @ beta_hat + omega_hat) 
            sigma_hat = np.mean(resid**2)
     
            params = ModelParameters(sigma=sigma_hat, omega=omega_hat,
                beta=beta_hat, phi=[], theta=[]) 
             
            llh = np.sum(self.calc_loglikelihood(params)) 
            k = np.sum(beta_hat != 0)  
            ic = ic_func(llh, k+1) # +1 for omega
       
            if ic < best_ic:
                best_ic = ic
                best_lambda = lam
                best_params = params
                best_resid = resid
                best_k = k
            
            if print_subresults:
                print(f"lambda {lam:.5f} | selected {k} | LLH {llh:.2f} | {criterion} {ic:.2f}")

            if k == 0:
                break
    
        lams = np.sort(np.array(lambda_grid))
    
        # store results
        self.params = best_params
        self.residuals = best_resid
        self.llh = np.sum(self.calc_loglikelihood(best_params))
        self.success = True
        
        tol = 1e-8
        zero_idx = np.where(np.abs(best_params.beta) < tol)[0]
        fixed_upd = self.fixed.copy()
        fixed_upd.beta[zero_idx] = True
    
        if print_results:
            print("\nFINAL:")
            print(f"lambda {best_lambda:.5f} | selected {best_k}") 
            self.summarize_results(title='LASSO estimation results estimation',fixed=fixed_upd)  
             
    def _build_fixed_g2s(self):
        """Boolean mask for beta parameters that cannot be dropped in G2S.
        True = protected (AR lags, seasonal dummies), False = eligible for dropping."""
        fixed_g2s = np.zeros(self.N_tot, dtype=bool)
        
        # protect seasonal dummies
        if self.bool_seas_dummies:
            fixed_g2s[:11] = True
        
        # protect AR lags
        if self.bool_include_AR:
            n_AR_lags = sum(len(lags) for lags in self.relevant_X_lags[:1])
            fixed_g2s[11*int(self.bool_seas_dummies): 11*int(self.bool_seas_dummies) + n_AR_lags] = True
        
        return fixed_g2s
    
    def _initialize_beta_OLS(self, params,fixed_beta):
        params_no_beta = params.copy()
        params_no_beta.beta[:] = 0
        not_fixed_beta = ~fixed_beta
        if self.num_comps > 0:
            residuals, _ = self.construct_filtered_seqs(params_no_beta)
        else:
            residuals = self.y.copy()
        X = self.X[:,not_fixed_beta]
        if not self.fixed.omega:
            X = sm.add_constant(X)
            if self.num_comps > 0:
                residuals += params.omega
        model = sm.regression.linear_model.OLS(residuals[self.burn:],X[self.burn:,:])
        results = model.fit()
        if not self.fixed.omega:
            params.omega = results.params[0]
        params.beta[not_fixed_beta] = results.params[int(1-self.fixed.omega):]
        return params
        
    def general_to_specific(self, update_self=True, params=None, fixed=None, 
                            sig_level=0.05, frac_drop=0.25, drop_bound=10):
        """General-to-specific lag/variable selection for beta."""
        if params is None:
            params = self.params
        if fixed is None:
            fixed = self.fixed
        params = params.copy()
        params_alt = params.copy()
        # use OLS to initialize beta's:
        if self.num_comps > 0:
            params_alt = self._initialize_beta_OLS(params_alt,fixed.beta)
        fixed = fixed.copy()
        
        fixed_g2s = self._build_fixed_g2s()
        # number of free, non-protected beta parameters
        free_X = np.sum(~fixed.beta & ~fixed_g2s) 
        
        crit = st.norm.ppf(1 - sig_level / 2)
        max_drop = max(int(np.round(free_X * frac_drop)), drop_bound)

        if free_X == 0:
            if self.num_comps > 0:
                params, llh, residuals, paths, success, _ = self._estimate_internal(params, fixed)
                std_errors = None
            else:
                params = self._initialize_beta_OLS(params, fixed.beta)
                residuals, paths = self.construct_filtered_seqs(params)
                residuals = residuals[self.burn:]
                paths = paths[:-1, :]
                params.sigma = np.mean(residuals ** 2)
                llh = np.sum(self.calc_loglikelihood(params))
                success = True
                std_errors = None

        for it in range(free_X): 
            # use OLS to initialize beta's:
            params = self._initialize_beta_OLS(params,fixed.beta)
            if self.num_comps > 0:
                params, llh, residuals, paths, success, _ = self._estimate_internal(params, fixed)
                params2, llh2, residuals2, paths2, success2, _ = self._estimate_internal(params_alt, fixed)
                if llh2 > llh or np.isnan(llh):
                    params, llh, residuals, paths, success = params2, llh2, residuals2, paths2, success2
            else:
                residuals, paths = self.construct_filtered_seqs(params)
                residuals = residuals[self.burn:]
                paths = paths[:-1,:]
                params.sigma = np.mean(residuals**2)
                llh = np.sum(self.calc_loglikelihood(params))
                success = True

            std_errors = self._calc_std_errors(params=params, fixed=fixed, robust=False)
            # calc t-stats
            beta_se = std_errors[-self.N_tot:] 
            t_stats = np.abs(params.beta / beta_se)
            
            # take out protected and already-fixed parameters
            t_stats_eligible = t_stats.copy() 
            t_stats_eligible[fixed_g2s | fixed.beta] = np.inf   # protected parameters never dropped/already dropped
            eligible_mask = ~fixed_g2s & ~fixed.beta
            has_insig = np.any(t_stats_eligible[eligible_mask] < crit)

            if has_insig:
                # drop insignificant, treat NaNs as significant (don't drop)
                t_stats_eligible[np.isnan(t_stats_eligible)] = np.inf
            else:
                # only NaNs or all significant: treat NaNs as insignificant
                nan_indices = np.where(np.isnan(t_stats_eligible) & eligible_mask)[0] 
                if len(nan_indices) > 0:
                    t_stats_eligible[nan_indices] = np.inf        # treat all as significant
                    t_stats_eligible[nan_indices[-1]] = 0.0       # except last one
            
            # check stopping condition
            n_insig = np.sum(t_stats_eligible < crit)
            if n_insig == 0:
                break
            
            # drop least significant
            n_drop = max(int(min(np.round(n_insig * frac_drop), max_drop)), 1)
            drop_idx = np.argpartition(t_stats_eligible, n_drop-1)[:n_drop]
            
            # fix dropped parameters at zero
            params.beta[drop_idx] = 0.0
            params_alt.beta[drop_idx] = 0.0
            fixed.beta[drop_idx] = True
        
        if update_self:
            self.params, self.llh, self.residuals, self.paths, self.success, self.fixed = \
                params, llh, residuals, paths, success, fixed
            self.std_errors = std_errors  
        
        return params, fixed
                
    def general_to_specific_grouped(self, params=None, fixed=None, sig_level=0.05,
                                     frac_drop=0.25, drop_bound=10, bool_random_search=False,
                                     print_progress=True, bool_pre_estimate=True, ):
        """General-to-specific with per-variable pre-selection followed by final G2S."""
        if params is None:
            params = self.params
        if fixed is None:
            fixed = self.fixed 
        params = params.copy()
          
        n_seas = 11 * int(self.bool_seas_dummies)
        n_AR = len(self.relevant_X_lags[0]) if self.bool_include_AR else 0
        n_protected = n_seas + n_AR
        
        if bool_pre_estimate and self.num_comps > 0:
            # estimate without X: fix all beta at zero
            fixed_pre = fixed.copy()
            fixed_pre.beta[n_protected:] = True
            params_pre = params.copy()
            params_pre.beta[n_protected:] = 0.0
            if bool_random_search:
                params, _, _, _, _ = self._random_start_search_internal(params_pre,fixed_pre,print_progress=False,n_draws=10000,n_best=10)   
            else:
                params, _, _, _, _, _ = self._estimate_internal(params_pre, fixed_pre)
        
        params_final_start = params.copy()
        
        # cumulative column indices per variable in beta (excluding protected)
        var_lags = self.relevant_X_lags[int(self.bool_include_AR):]
        cumsum = np.zeros(len(var_lags) + 1, dtype=int)
        for i, lags in enumerate(var_lags):
            cumsum[i+1] = cumsum[i] + len(lags)
    
        # per-variable G2S pre-selection
        start_tot = timeit.default_timer()
        for i, lags in enumerate(var_lags):
            if print_progress:
                start = timeit.default_timer()
                print(f"Variable {i+1}/{len(var_lags)}")
        
            var_beta_idx = np.arange(n_protected + cumsum[i], n_protected + cumsum[i+1])
        
            fixed_var = fixed.copy()
            fixed_var.beta[n_protected:] = True
            fixed_var.beta[var_beta_idx] = fixed.beta[var_beta_idx]
        
            params_var = params.copy()
            params_var.beta[n_protected:n_protected + cumsum[i]] = 0.0
            params_var.beta[n_protected + cumsum[i+1]:] = 0.0
        
            params_var, fixed_var = self.general_to_specific(params=params_var, fixed=fixed_var, 
                                                   sig_level=sig_level, frac_drop=frac_drop,  
                                                   drop_bound=drop_bound, update_self=False)
            fixed.beta[var_beta_idx] = fixed_var.beta[var_beta_idx]
            #params.beta[var_beta_idx] = params_var.beta[var_beta_idx] 
        
            if print_progress:
                stop = timeit.default_timer()
                n_selected = np.sum(~fixed.beta[var_beta_idx])
                print(f"  Selected {n_selected}/{len(lags)} lags, time: {stop-start:.2f}s")    
        
        if print_progress:
            n_selected_total = np.sum(~fixed.beta[n_protected:])
            print(f"Selected before final G2S: {n_selected_total}/{cumsum[-1]}")
    
        # final G2S on all selected variables
        params_final_start.beta[np.where(fixed.beta)] = 0.0
        params, fixed_upd = self.general_to_specific(params=params_final_start, fixed=fixed, sig_level=sig_level,
                         frac_drop=frac_drop, drop_bound=drop_bound, update_self=True)
        stop_tot = timeit.default_timer()
        if print_progress:
            print(f"Total time: {stop_tot - start_tot:.2f}s")
            self.summarize_results(title='G2S estimation results estimation',fixed=fixed_upd)  
        
    def _draw_random_params(self, fixed=None, params=None):
        if fixed is None:
            fixed = self.fixed
        if params is None:
            params = self.params
        
        params = params.copy()
        b = self.search_bounds
        
        # sigma
        if not fixed.sigma and 'sigma' in b:
            params.sigma = np.random.uniform(b['sigma'][0], b['sigma'][1] * np.var(self.y))
        
        # phi and theta
        for k in range(self.num_comps):
            phi_bounds = b.get(f'phi_{k+1}')
            theta_bounds = b.get(f'theta_{k+1}')
            
            # draw phi_2 first for AR(2) components
            if self.vec_autoreg_lags[k] == 2:
                if not fixed.phi[k][1] and phi_bounds is not None and 1 in phi_bounds:
                    params.phi[k][1] = np.random.uniform(*phi_bounds[1])
            
            # draw phi_1 and remaining lags
            for j in range(self.vec_autoreg_lags[k]):
                if j == 1 and self.vec_autoreg_lags[k] == 2:
                    continue
                if not fixed.phi[k][j] and phi_bounds is not None and j in phi_bounds:
                    if j == 0 and self.vec_autoreg_lags[k] == 2:
                        if k+1 in self.cycle_per_restrictions:
                            if self.cycle_per_restrictions[k+1] is not None:
                                bounds = self.cycle_per_restrictions[k+1]
                                phi1_min = 2 * np.sqrt(-params.phi[k][1]) * np.cos(2 * np.pi / bounds[0])
                                phi1_max = 2 * np.sqrt(-params.phi[k][1]) * np.cos(2 * np.pi / bounds[1])
                                params.phi[k][0] = np.random.uniform(min(phi1_min, phi1_max), max(phi1_min, phi1_max))
                            else:
                                rho = np.random.uniform(0, 1)
                                params.phi[k][0] = 2 * np.sqrt(-params.phi[k][1]) * rho
                        else:
                            params.phi[k][0] = np.random.uniform(*phi_bounds[0])
                    else:
                        params.phi[k][j] = np.random.uniform(*phi_bounds[j])
                elif fixed.phi[k][j] and self.vec_autoreg_lags[k] == 2 and j == 0:
                    if k+1 in self.cycle_per_restrictions and self.cycle_per_restrictions[k+1] is not None:
                        target_period = np.mean(self.cycle_per_restrictions[k+1])
                        params.phi[k][0] = 2 * np.sqrt(-params.phi[k][1]) * np.cos(2 * np.pi / target_period)
            
            # theta
            for j in range(self.vec_ma_lags[k]):
                if not fixed.theta[k][j] and theta_bounds is not None and j in theta_bounds:
                    params.theta[k][j] = np.random.uniform(*theta_bounds[j])
        
        # beta
        n_seas = 11 * int(self.bool_seas_dummies)
        for j in range(self.N_tot):
            if not fixed.beta[j]:
                if j < n_seas:
                    if 'beta_seas' in b:
                        params.beta[j] = np.random.uniform(*b['beta_seas'])
                else:
                    if 'beta' in b:
                        params.beta[j] = np.random.uniform(*b['beta'])
        
        return params
            
    def _random_start_search_internal(self, params, fixed, n_draws=5000, n_best=5, print_progress=True):
        llh_vals = []
        param_draws = []
        
        for idx, _ in enumerate(range(n_draws)):
            if print_progress and idx % 1000 == 0:
                print(f"Random search: {idx}/{n_draws} draws, {len(llh_vals)} finite")
            params = self._draw_random_params(params=params,fixed=fixed)
            llh = np.mean(self.calc_loglikelihood(params))
            if np.isfinite(llh):
                llh_vals.append(llh)
                param_draws.append(params)
              
        if len(llh_vals) == 0:
            print("Warning: no finite likelihood values found in random search")
            return
            
        # optimize from n_best starting points
        best_llh = -np.inf
        for rank, i in enumerate(np.argsort(llh_vals)[-n_best:]):
            if print_progress:
                print(f"Optimizing from starting point {rank+1}/{n_best}")
            params, llh, residuals, paths, success, _ = self._estimate_internal(param_draws[i], fixed)
            if llh > best_llh:
                best_llh = llh
                best_params, best_residuals, best_paths, best_success = params, residuals, paths, success
         
        return  best_params, best_llh, best_residuals, best_paths, best_success     
    
    def random_start_search(self, n_draws=5000, n_best=5, print_progress=True,construct_std_errors = True,check_hess_posdef = True):
        self.params, self.llh, self.residuals, self.paths, \
                  self.success = self._random_start_search_internal(self.params, self.fixed,n_draws=n_draws,n_best=n_best,print_progress=print_progress)
        if construct_std_errors:
            self.std_errors = self._calc_std_errors(robust=False, check_hess_posdef=check_hess_posdef)
         
    def construct_forecast(self, params=None):
        """ Construct horizon-step ahead forecast""" 
        if params is None:
            params = self.params  
        
        if self.N -int(self.bool_include_AR) > 0 and self.X_full.shape[0] < self.T_insample_fin + self.horizon:
            raise ValueError("Not enough X observations for forecasting")
         
        X = None
        if self.N + int(self.bool_seas_dummies) -int(self.bool_include_AR) > 0:
            X=self.X_full[:self.T_insample_fin + self.horizon, :]
            
        if self.num_comps == 0 and not self.bool_include_AR: 
            if self.N_tot == 0:
                return params.omega
            else:
                return float(X[-1,:].reshape(1,-1) @ params.beta + params.omega)
        if self.bool_include_AR: 
            n_seas_dummies = self.bool_seas_dummies * 11 
            ar_lags = len(self.relevant_X_lags[0]) # code requires all lags from 1 to p to be included
            yhat = np.zeros(ar_lags + self.horizon)
            yhat[:ar_lags] = self.y[-ar_lags:]
            ar_params = np.flip(params.beta[n_seas_dummies:n_seas_dummies+ar_lags])
            
            for h in range(self.horizon):
                yhat[ar_lags + h] =  params.omega  + np.sum(ar_params * yhat[h:ar_lags + h]) 
                if self.bool_seas_dummies:
                    yhat[ar_lags + h] = yhat[ar_lags + h] + np.sum(params.beta[:n_seas_dummies] * X[-self.horizon + h,:n_seas_dummies])
                if self.N>1:
                    yhat[ar_lags + h] = yhat[ar_lags + h] + np.sum(params.beta[n_seas_dummies+ar_lags:] * X[-self.horizon + h,n_seas_dummies+ ar_lags:]) 
            return float(yhat[-1])
     
        residuals, paths = self.construct_filtered_seqs(params) 
        
        max_ar_lag = np.max(self.vec_autoreg_lags)
        max_ma_lag = np.max(self.vec_ma_lags)
         
        mat_phi = np.zeros((max_ar_lag, self.num_comps))
        mat_theta = np.zeros((max_ma_lag, self.num_comps)) 
        for i in range(self.num_comps):
            mat_phi[-self.vec_autoreg_lags[i]:, i] = np.flip(params.phi[i])
            mat_theta[-self.vec_ma_lags[i]:, i] = np.flip(params.theta[i]) 
            
        eps_ext = np.concatenate([residuals, np.zeros(self.horizon)])
        paths_ext = np.vstack([paths, np.zeros((self.horizon-1, self.num_comps))])
        
        for t in range(self.T_insample_fin+1, self.T_insample_fin + self.horizon):
            for j in range(self.num_comps):
                val = 0.0
                # AR part
                for k in range(max_ar_lag):
                    val += paths_ext[t-max_ar_lag+k, j] * mat_phi[k, j]
                # MA part
                for k in range(max_ma_lag):
                    val += mat_theta[k, j] * eps_ext[t-max_ma_lag+k]
                paths_ext[t, j] = val 
        
        forecast = np.sum(paths_ext[-1, :]) + params.omega 
        if self.N_tot > 0:
            forecast += np.dot(params.beta, X[-1, :])
        
        return float(forecast)
     
    def calc_freqs(self): 
        freq_dict = {}
        
        for i in range(self.num_comps):
            if self.vec_autoreg_lags[i] == 2:
                phi1 = self.params.phi[i][0]
                phi2 = self.params.phi[i][1]
                
                roots = np.roots([1, -phi1, -phi2])
                modulus = np.abs(roots[0])
                
                term = phi1 /(2 * np.sqrt(-phi2))
                if np.abs(term) < 1:
                    freq =  math.acos(term)/(2*np.pi) 
                    period = 1/freq
                else :
                    freq = None
                    period = None
                 
                freq_dict[i+1] = {
                'freq': freq,
                'period': period,
                'roots': roots,
                'modulus': modulus,
                'is_cyclical': freq is not None
                }
            
        return freq_dict
                 
    def summarize_results(self, print_results = True, bool_plot_filt_vals = True, title='Estimation results', fixed = None):
        """
        Construct and print a summary table of parameter estimates,
        standard errors, and model fit statistics.
        """
        if bool_plot_filt_vals:
            self.plot_filt_values()
        
        if fixed is None:
            fixed = self.fixed
        
        flat_params = self._params_to_vector(self.params)
        
        if self.std_errors is None:
            std_errs = np.full_like(flat_params, np.nan, dtype=float) 
        else:
            std_errs = self.std_errors
              
        df = pd.DataFrame({"Param": self.tot_labels,
            "Est": flat_params, "Std. error": std_errs })
        
        num_free = self._get_no_free_params(fixed=fixed)
        aic = -2 * self.llh + 2 * num_free 
        bic = -2 * self.llh + num_free * np.log(self.T_insample_fin-self.burn) 
        
        freq_dict = self.calc_freqs()
        # add cycle periods to parameter table
        period_rows = []
        for k, info in freq_dict.items():
            if info['is_cyclical']:
                period_rows.append({
                    "Param": f"period_{k}",
                    "Est": info['period'],
                    "Std. error": np.nan
                })
        
        stats = pd.DataFrame({
            "Param": ["LLH", "AIC", "BIC"],
            "Est": [self.llh, aic, bic],
            "Std. error": [np.nan, np.nan, np.nan]})
        
        n_seas = 11 * int(self.bool_seas_dummies)
        n_AR = len(self.relevant_X_lags[0]) if self.bool_include_AR else 0
        n_protected = n_seas + n_AR
        tol = 1e-8
        k_beta = np.sum(np.abs(self.params.beta[n_protected:]) > tol)
        
        extra = pd.DataFrame({
            "Param": ["no. selected vars", "total vars"],
            "Est": [k_beta, self.N_tot-n_AR-n_seas],
            "Std. error": [np.nan, np.nan]
        })
        
        df_tot = pd.concat([df, stats,pd.DataFrame(period_rows),extra], ignore_index=True) 
        
        # ---- print results ----
        if print_results:  
            # for beta, also filter out fixed-at-zero parameters (dropped in G2S/LASSO)
            n_non_beta = len(flat_params) - self.N_tot
            beta_mask = np.zeros(len(flat_params), dtype=bool)
            beta_mask[n_non_beta:] = fixed.beta  # fixed beta entries 
            print_mask = ~beta_mask  # show all non-beta params + non-fixed betas
            
            df_print = pd.DataFrame({"Param": self.tot_labels[print_mask],
                "Est": flat_params[print_mask], "Std. error": std_errs[print_mask] })
            
            print("\n" + "=" * len(title))
            print(title)
            print("=" * len(title))
            print(df_print.fillna("–").to_string(index=False, justify="left", col_space=10, float_format=lambda x: f"{x:.4f}"))
            print("\n" + "=" * len(title) )
            print(stats.fillna(" ").to_string(index=False, justify="left", col_space=5, header=False, float_format=lambda x: f"{x:.3f}"))

        # ---- cycle frequencies ----  
            if freq_dict:
                print("\n" + "=" * len(title))
                print("Cycle diagnostics")
                print("=" * len(title))
                for k, info in freq_dict.items():
                    cyclical_str = f"period = {info['period']:.3f}, freq = {info['freq']:.4f}" if info['is_cyclical'] else "real roots"
                    print(f"  Component {k}: {cyclical_str}, roots = ({info['roots'][0]:.4f}, {info['roots'][1]:.4f}), modulus = {info['modulus']:.4f}")

        # ---- number of selected vars ----  
            if self.N_tot > 0:
                print("\n" + "=" * len(title))
                print("Number of selected vars")
                print("=" * len(title))
                print(extra.fillna(" ").to_string(index=False, justify="left", header=False, col_space=10))

        return df_tot 
        
    def plot_filt_values(self):
        """Plot the data and the filtered components."""
 
        index = self.dates_insample if self.dates_insample is not None else np.arange(1, self.T_insample_fin + 1)
 
        bool_dates = self.dates_insample is not None
        
        def plot(index, *args, **kwargs):
            if bool_dates:
                plt.plot_date(index, *args, fmt='-', **kwargs)
            else:
                plt.plot(index, *args, **kwargs)
    
        # plot data and first component/constant
        line = self.paths[:, 0] if self.bool_trend else np.ones_like(self.y) * self.params.omega
        if self.num_comps > 0:
            self.paths[:, 0] = line
        comp1_label = rf'$\hat{{f}}^{{(1)}}_{{t}}$' if self.bool_trend else 'constant'

        plt.figure() 
        plot(index, self.y, label='data')
        plot(index, line, label=comp1_label)
        plt.title("Data and " + ("Component 1" if self.bool_trend else "Constant"))
        plt.legend()
        plt.show()
    
        # plot remaining components
        for i in range(int(self.bool_trend), self.num_comps): 
            plt.figure()
            plot(index, self.paths[:, i], label=rf'$\hat{{f}}^{{({i+1})}}_{{t}}$')
            if np.min(self.paths[:, i]) < 0 < np.max(self.paths[:, i]):
                plt.axhline(color='k')
            plt.title(f"Component {i+1}")
            plt.legend()
            plt.show()  