# ======================================================================================================================
"""
This config file defines the models and sub-samples for Tables 3, A.8, A.9, A.10, and A.11.

By Sebastian Jensen
Aarhus University
CoRE
Jul, 2026
"""
# ======================================================================================================================

# Models for the main text (Table 3)
Main = [
    '0componentsSettingA',
    'G2S0componentsSettingAwithExog', 'G2S0componentsSettingAwithExogPCA',
    'Lasso0componentsSettingAwithExog', 'Lasso0componentsSettingAwithExogPCA',
    '3componentsSettingB', '3componentsSettingC',
    '5componentsSettingB', '5componentsSettingC',
    'G2S5componentsSettingBwithExog', 'G2S5componentsSettingBwithExogPCA',
    '0componentsSettingA_AR1_anomalies', '0componentsSettingA_AR_anomalies',
    'G2S0componentsSettingA_AR1_anomalieswithExog','G2S0componentsSettingA_AR1_anomalieswithExogPCA',
    'G2S0componentsSettingA_AR_anomalieswithExog', 'G2S0componentsSettingA_AR_anomalieswithExogPCA'
]

# Models for the Appendix (Table A.8)
Appendix = [
    '5componentsSettingB',
    '3componentsSettingB_seasdummies',
    '5componentsSettingB_seasdummies',
    'G2S5componentsSettingBwithExog',
    'G2S3componentsSettingB_seasdummieswithExog',
    'G2S5componentsSettingB_seasdummieswithExog',
    'G2S5componentsSettingBwithExogPCA',
    'G2S3componentsSettingB_seasdummieswithExogPCA',
    'G2S5componentsSettingB_seasdummieswithExogPCA'
]

# El Nino periods (Table A.9)
Nino = [
    '2005-01', '2005-02', '2005-03',
    '2006-08', '2006-09', '2006-10', '2006-11', '2006-12', '2007-01', '2007-02',
    '2009-06', '2009-07', '2009-08', '2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02', '2010-03', '2010-04',
    '2014-09', '2014-10', '2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03', '2016-04', '2016-05',
    '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07',
    '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05'
]

# La Nina periods (Table A.10)
Nina = [
    '2005-10', '2005-11', '2005-12', '2006-01', '2006-02', '2006-03', '2006-04',
    '2007-05', '2007-06', '2007-07', '2007-08', '2007-09', '2007-10', '2007-11', '2007-12', '2008-01', '2008-02', '2008-03', '2008-04', '2008-05', '2008-06', '2008-07',
    '2008-10', '2008-11', '2008-12', '2009-01', '2009-02', '2009-03', '2009-04',
    '2010-05', '2010-06', '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05',
    '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12', '2017-01',
    '2017-09', '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05',
    '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02'
]

# Models without anomalies benchmarks (Table A.11)
Score_nams = [
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
]

# Test sample
test_sample = [f"{year}-{month:02d}" for year in range(2005, 2025) for month in range(1, 13)]

# Forecasting horizons
horizons = [1,3,6,12,14,16,18]

# Variable names mapping
rename_map = {
    '0componentsSettingA': 'Constant',
    '3componentsSettingB': 'Constant + seasonals',
    '3componentsSettingB_seasdummies': 'Constant + dummies + cycles',
    '3componentsSettingC': 'Trend + seasonals',
    '5componentsSettingB': 'Constant + seasonals + cycles',
    '5componentsSettingB_seasdummies': 'Constant + seasonals + dummies + cycles',
    '5componentsSettingC': 'Trend + seasonals + cycles',
    'G2S0componentsSettingAwithExog': 'Regressors: G2S',
    'G2S0componentsSettingAwithExogPCA': 'Regressors: G2S PCA',
    'Lasso0componentsSettingAwithExog': 'Regressors: LASSO',
    'Lasso0componentsSettingAwithExogPCA': 'Regressors: LASSO PCA',
    'G2S3componentsSettingB_seasdummieswithExog': 'Constant + dummies + cycles + G2S regressors',
    'G2S3componentsSettingB_seasdummieswithExogPCA': 'Constant + dummies + cycles + G2S PCA regressors',
    'G2S5componentsSettingBwithExog': 'Constant + seasonals + cycles + G2S regressors',
    'G2S5componentsSettingB_seasdummieswithExog': 'Constant + seasonals + dummies + cycles + G2S regressors',
    'G2S5componentsSettingBwithExogPCA': 'Constant + seasonals + cycles + G2S PCA regressors',
    'G2S5componentsSettingB_seasdummieswithExogPCA': 'Constant + seasonals + dummies + cycles + G2S PCA regressors',
    '0componentsSettingA_AR_anomalies': 'Anomalies: AR(3)',
    'G2S0componentsSettingA_AR_anomalieswithExog': 'Anomalies: ARX(3) G2S',
    'G2S0componentsSettingA_AR_anomalieswithExogPCA': 'Anomalies: ARX(3) G2S PCA',
    '0componentsSettingA_AR1_anomalies': 'Anomalies: AR(1)',
    'G2S0componentsSettingA_AR1_anomalieswithExog': 'Anomalies: ARX(1) G2S',
    'G2S0componentsSettingA_AR1_anomalieswithExogPCA': 'Anomalies: ARX(1) G2S PCA'
}

