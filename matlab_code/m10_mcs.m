% =========================================================================
% This script apllies the model confidence set (MCS) to the ENSO forecasts 
% and is used to create Tables 3, A.8, A.9, A.10, and A.11.
%
% Required toolboxes:
%   mfe-toolbox
% 
% By Sebastian Jensen
% Aarhus University
% CoRE
% Jul, 2026
% =========================================================================

%% MATLAB environment
clear;
close all;
clc;
scriptDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(scriptDir, 'mfe-toolbox-main', 'mfe-toolbox-main', 'bootstrap')));

rng(123);  % random seed


%% MCS choice parameters
B = 100000;             % n.o. bootstraps
BOOT = 'STATIONARY';    % bootstrap method
W = 9;                  % block length


%% Experimental setup
hs = [1, 3, 6, 12, 14, 16, 18];  % forecast horizons

% Models for the main text (Table 3)
main = {
    '0componentsSettingA',
    'G2S0componentsSettingAwithExog', 
    'G2S0componentsSettingAwithExogPCA',
    'Lasso0componentsSettingAwithExog', 
    'Lasso0componentsSettingAwithExogPCA',
    '3componentsSettingB', 
    '3componentsSettingC',
    '5componentsSettingB', 
    '5componentsSettingC',
    'G2S5componentsSettingBwithExog', 
    'G2S5componentsSettingBwithExogPCA',
    '0componentsSettingA_AR1_anomalies'
    '0componentsSettingA_AR_anomalies', 
    'G2S0componentsSettingA_AR1_anomalieswithExog',
    'G2S0componentsSettingA_AR1_anomalieswithExogPCA',
    'G2S0componentsSettingA_AR_anomalieswithExog',
    'G2S0componentsSettingA_AR_anomalieswithExogPCA'
    };

% Models for the Appendix (Table A.8)
appendix = {
    '5componentsSettingB',
    '3componentsSettingB_seasdummies',
    '5componentsSettingB_seasdummies',
    'G2S5componentsSettingBwithExog',
    'G2S3componentsSettingB_seasdummieswithExog',
    'G2S5componentsSettingB_seasdummieswithExog',
    'G2S5componentsSettingBwithExogPCA',
    'G2S3componentsSettingB_seasdummieswithExogPCA',
    'G2S5componentsSettingB_seasdummieswithExogPCA'
    };

% El Nino periods (Table A.9)
Nino = {
    '2005-01-01'; '2005-02-01'; '2005-03-01';...
    '2006-08-01'; '2006-09-01'; '2006-10-01'; '2006-11-01'; '2006-12-01'; '2007-01-01'; '2007-02-01';...
    '2009-06-01'; '2009-07-01'; '2009-08-01'; '2009-09-01'; '2009-10-01'; '2009-11-01'; '2009-12-01'; '2010-01-01'; '2010-02-01'; '2010-03-01'; '2010-04-01';...
    '2014-09-01'; '2014-10-01'; '2014-11-01'; '2014-12-01'; '2015-01-01'; '2015-02-01'; '2015-03-01'; '2015-04-01'; '2015-05-01'; '2015-06-01'; '2015-07-01'; '2015-08-01'; '2015-09-01'; '2015-10-01'; '2015-11-01'; '2015-12-01'; '2016-01-01'; '2016-02-01'; '2016-03-01'; '2016-04-01'; '2016-05-01';...
    '2018-08-01'; '2018-09-01'; '2018-10-01'; '2018-11-01'; '2018-12-01'; '2019-01-01'; '2019-02-01'; '2019-03-01'; '2019-04-01'; '2019-05-01'; '2019-06-01'; '2019-07-01';...
    '2023-04-01'; '2023-05-01'; '2023-06-01'; '2023-07-01'; '2023-08-01'; '2023-09-01'; '2023-10-01'; '2023-11-01'; '2023-12-01'; '2024-01-01'; '2024-02-01'; '2024-03-01'; '2024-04-01'; '2024-05-01'...
    };

% La Nina periods (Table A.10)
Nina = {
    '2005-10-01', '2005-11-01', '2005-12-01', '2006-01-01', '2006-02-01', '2006-03-01', '2006-04-01',...
    '2007-05-01', '2007-06-01', '2007-07-01', '2007-08-01', '2007-09-01', '2007-10-01', '2007-11-01', '2007-12-01', '2008-01-01', '2008-02-01', '2008-03-01', '2008-04-01', '2008-05-01', '2008-06-01', '2008-07-01',...
    '2008-10-01', '2008-11-01', '2008-12-01', '2009-01-01', '2009-02-01', '2009-03-01', '2009-04-01',...
    '2010-05-01', '2010-06-01', '2010-07-01', '2010-08-01', '2010-09-01', '2010-10-01', '2010-11-01', '2010-12-01', '2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01', '2011-05-01', '2011-06-01', '2011-07-01', '2011-08-01', '2011-09-01', '2011-10-01', '2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01', '2012-03-01', '2012-04-01', '2012-05-01',...
    '2016-07-01', '2016-08-01', '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01', '2017-01-01',...
    '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01', '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01',...
    '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01'...
    };

% Models without anomalies benchmarks (Table A.11)
Score_nams = {
    '0componentsSettingA',
    'G2S0componentsSettingAwithExog', 
    'G2S0componentsSettingAwithExogPCA',
    'Lasso0componentsSettingAwithExog', 
    'Lasso0componentsSettingAwithExogPCA',
    '3componentsSettingB', 
    '3componentsSettingC',
    '5componentsSettingB', 
    '5componentsSettingC',
    'G2S5componentsSettingBwithExog', 
    'G2S5componentsSettingBwithExogPCA'
    };

% Variable names mapping
rename_map = containers.Map( ...
    { ...
    '0componentsSettingA', ...
    '3componentsSettingB', ...
    '3componentsSettingB_seasdummies', ...
    '3componentsSettingC', ...
    '5componentsSettingB', ...
    '5componentsSettingB_seasdummies', ...
    '5componentsSettingC', ...
    'G2S0componentsSettingAwithExog', ...
    'G2S0componentsSettingAwithExogPCA', ...
    'Lasso0componentsSettingAwithExog', ...
    'Lasso0componentsSettingAwithExogPCA', ...
    'G2S3componentsSettingB_seasdummieswithExog', ...
    'G2S3componentsSettingB_seasdummieswithExogPCA', ...
    'G2S5componentsSettingBwithExog', ...
    'G2S5componentsSettingB_seasdummieswithExog', ...
    'G2S5componentsSettingBwithExogPCA', ...
    'G2S5componentsSettingB_seasdummieswithExogPCA', ...
    '0componentsSettingA_AR_anomalies', ...
    'G2S0componentsSettingA_AR_anomalieswithExog', ...
    'G2S0componentsSettingA_AR_anomalieswithExogPCA', ...
    '0componentsSettingA_AR1_anomalies', ...
    'G2S0componentsSettingA_AR1_anomalieswithExog', ...
    'G2S0componentsSettingA_AR1_anomalieswithExogPCA' ...
    }, ...
    {...
    'Constant', ...
    'Constant + seasonals', ...
    'Constant + dummies + cycles', ...
    'Trend + seasonals', ...
    'Constant + seasonals + cycles', ...
    'Constant + seasonals + dummies + cycles', ...
    'Trend + seasonals + cycles', ...
    'Regressors: G2S', ...
    'Regressors: G2S PCA', ...
    'Regressors: LASSO', ...
    'Regressors: LASSO PCA', ...
    'Constant + dummies + cycles + G2S regressors', ...
    'Constant + dummies + cycles + G2S PCA regressors', ...
    'Constant + seasonals + cycles + G2S regressors', ...
    'Constant + seasonals + dummies + cycles + G2S regressors', ...
    'Constant + seasonals + cycles + G2S PCA regressors', ...
    'Constant + seasonals + dummies + cycles + G2S PCA regressors', ...
    'Anomalies: AR(3)', ...
    'Anomalies: ARX(3) G2S', ...
    'Anomalies: ARX(3) G2S PCA', ...
    'Anomalies: AR(1)', ...
    'Anomalies: ARX(1) G2S', ...
    'Anomalies: ARX(1) G2S PCA' ...
    }...
    );


%% Constructing model cofidence sets for the main text (Table 3)
models = main;

% Table to store MCS results
mcs_results = strings(numel(models), numel(hs));
model_names = {};

for j = 1:numel(hs)
    h = hs(j);

    % Load forecast errors
    errors = readtable(append('../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon',num2str(h),'.csv'), delimiter=',', ReadRowNames=true, VariableNamingRule='preserve');
    
    % Remove horizon-specifc suffix from variable names
    errors.Properties.VariableNames = strrep(errors.Properties.VariableNames, ['horizon' num2str(h)], '');

    % Extract models
    [model_found, model_idx] = ismember(models, errors.Properties.VariableNames);

    losses_tab = errors(:, model_idx(model_found)).^2;
    losses_mat = table2array(losses_tab);

    % Renaming variables
    old = losses_tab.Properties.VariableNames;
    old = old(ismember(old, keys(rename_map)));
    
    new = values(rename_map, old);
    
    losses_tab = renamevars(losses_tab, old, new);

    % Store model names
    if isempty(model_names)
        model_names = losses_tab.Properties.VariableNames;
    end

    % MCS at alpha = 0.10
    [included_10,~,~,~,~,~] = mcs(losses_mat, 0.10, B, W, BOOT);
    mcs_results(included_10, j) = "*";

    % MCS at alpha = 0.25
    [included_25,~,~,~,~,~] = mcs(losses_mat, 0.25, B, W, BOOT);
    mcs_results(included_25, j) = "**";
end

% Convert results to a table
mcs_table = array2table(mcs_results, VariableNames=string(hs), RowNames=model_names);
disp("Table 3")
disp(mcs_table)


%% Constructing model cofidence sets for the Appendix (Table A.8)
models = appendix;

% Table to store MCS results
mcs_results = strings(numel(models), numel(hs));
model_names = {};

for j = 1:numel(hs)
    h = hs(j);
    
    % Load forecast errors
    errors = readtable(append('../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon',num2str(h),'.csv'), delimiter=',', ReadRowNames=true, VariableNamingRule='preserve');

    % Remove horizon-specifc suffix from variable names
    errors.Properties.VariableNames = strrep(errors.Properties.VariableNames, ['horizon' num2str(h)], '');

    % Extract models
    [model_found, model_idx] = ismember(models, errors.Properties.VariableNames);

    losses_tab = errors(:, model_idx(model_found)).^2;
    losses_mat = table2array(losses_tab);

    % Renaming variables
    old = losses_tab.Properties.VariableNames;
    old = old(ismember(old, keys(rename_map)));
    
    new = values(rename_map, old);
    
    losses_tab = renamevars(losses_tab, old, new);

    % Store model names
    if isempty(model_names)
        model_names = losses_tab.Properties.VariableNames;
    end

    % MCS at alpha = 0.10
    [included_10,~,~,~,~,~] = mcs(losses_mat, 0.10, B, W, BOOT);
    mcs_results(included_10, j) = "*";

    % MCS at alpha = 0.25
    [included_25,~,~,~,~,~] = mcs(losses_mat, 0.25, B, W, BOOT);
    mcs_results(included_25, j) = "**";
end

% Convert results to a table
mcs_table = array2table(mcs_results, VariableNames=string(hs), RowNames=model_names);
disp("Table A.8")
disp(mcs_table)


%% Constructing model cofidence sets for El Nino periods (Table A.9)
models = main;

% Table to store MCS results
mcs_results = strings(numel(models), numel(hs));
model_names = {};

for j = 1:numel(hs)
    h = hs(j);

    % Load forecast errors
    errors = readtable(append('../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon',num2str(h),'.csv'), delimiter=',', ReadRowNames=true, VariableNamingRule='preserve');

    % Remove horizon-specifc suffix from variable names
    errors.Properties.VariableNames = strrep(errors.Properties.VariableNames, ['horizon' num2str(h)], '');
    
    % Extract models
    [model_found, model_idx] = ismember(models, errors.Properties.VariableNames);

    losses_tab = errors(:, model_idx(model_found)).^2;

    % Renaming variables
    old = losses_tab.Properties.VariableNames;
    old = old(ismember(old, keys(rename_map)));
    
    new = values(rename_map, old);
    
    losses_tab = renamevars(losses_tab, old, new);

    % Store model names
    if isempty(model_names)
        model_names = losses_tab.Properties.VariableNames;
    end

    % Extract Nino periods
    Nino_events = ismember(errors.Properties.RowNames, Nino);
    losses_tab = losses_tab(Nino_events,:);
    losses_mat = table2array(losses_tab);

    % MCS at alpha = 0.10
    [included_10,~,~,~,~,~] = mcs(losses_mat, 0.10, B, W, BOOT);
    mcs_results(included_10, j) = "*";

    % MCS at alpha = 0.25
    [included_25,~,~,~,~,~] = mcs(losses_mat, 0.25, B, W, BOOT);
    mcs_results(included_25, j) = "**";
end

% Convert results to a table
mcs_table = array2table(mcs_results, VariableNames=string(hs), RowNames=model_names);
disp("Table A.9")
disp(mcs_table)


%% Constructing model cofidence sets for La Nina periods (Table A.10)
models = main;

% Table to store MCS results
mcs_results = strings(numel(models), numel(hs));
model_names = {};

for j = 1:numel(hs)
    h = hs(j);
    
    % Load forecast errors
    errors = readtable(append('../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon',num2str(h),'.csv'), delimiter=',', ReadRowNames=true, VariableNamingRule='preserve');

    % Remove horizon-specifc suffix from variable names
    errors.Properties.VariableNames = strrep(errors.Properties.VariableNames, ['horizon' num2str(h)], '');
    
    % Extract models
    [model_found, model_idx] = ismember(models, errors.Properties.VariableNames);

    losses_tab = errors(:, model_idx(model_found)).^2;

    % Renaming variables
    old = losses_tab.Properties.VariableNames;
    old = old(ismember(old, keys(rename_map)));
    
    new = values(rename_map, old);
    
    losses_tab = renamevars(losses_tab, old, new);

    % Store model names
    if isempty(model_names)
        model_names = losses_tab.Properties.VariableNames;
    end

    % Extract Nina periods
    Nina_events = ismember(errors.Properties.RowNames, Nina);
    losses_tab = losses_tab(Nina_events,:);
    losses_mat = table2array(losses_tab);

    % MCS at alpha = 0.10
    [included_10,~,~,~,~,~] = mcs(losses_mat, 0.10, B, W, BOOT);
    mcs_results(included_10, j) = "*";

    % MCS at alpha = 0.25
    [included_25,~,~,~,~,~] = mcs(losses_mat, 0.25, B, W, BOOT);
    mcs_results(included_25, j) = "**";
end

% Convert results to a table
mcs_table = array2table(mcs_results, VariableNames=string(hs), RowNames=model_names);
disp("Table A.10")
disp(mcs_table)


%% Constructing model cofidence sets without anomalies benchmarks (Table A.11)
models = Score_nams;

% Table to store MCS results
mcs_results = strings(numel(models), numel(hs));
model_names = {};

for j = 1:numel(hs)
    h = hs(j);
    
    % Load forecast errors
    errors = readtable(append('../output/intermediary_data/outsample/combined_forecast_errors/combined_forecast_errors_horizon',num2str(h),'.csv'), delimiter=',', ReadRowNames=true, VariableNamingRule='preserve');

    % Remove horizon-specifc suffix from variable names
    errors.Properties.VariableNames = strrep(errors.Properties.VariableNames, ['horizon' num2str(h)], '');
    
    % Extract models
    [model_found, model_idx] = ismember(models, errors.Properties.VariableNames);

    losses_tab = errors(:, model_idx(model_found)).^2;
    losses_mat = table2array(losses_tab);

    % Renaming variables
    old = losses_tab.Properties.VariableNames;
    old = old(ismember(old, keys(rename_map)));
    
    new = values(rename_map, old);
    
    losses_tab = renamevars(losses_tab, old, new);

    % Store model names
    if isempty(model_names)
        model_names = losses_tab.Properties.VariableNames;
    end

    % MCS at alpha = 0.10
    [included_10,~,~,~,~,~] = mcs(losses_mat, 0.10, B, W, BOOT);
    mcs_results(included_10, j) = "*";

    % MCS at alpha = 0.25
    [included_25,~,~,~,~,~] = mcs(losses_mat, 0.25, B, W, BOOT);
    mcs_results(included_25, j) = "**";
end

% Convert results to a table
mcs_table = array2table(mcs_results, VariableNames=string(hs), RowNames=model_names);
disp("Table A.11")
disp(mcs_table)

