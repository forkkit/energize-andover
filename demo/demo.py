#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:14:27 2017

@author: matt
"""

import energize as egz
import pandas as pd

# IMPORT THE TREND DATA
df_trend = pd.read_csv('trend.csv', engine='python', index_col=0, parse_dates=[0])

# MAKE SURE IT'S AT A CONSTANT FREQUENCY
df_trend = df_trend.dropna().asfreq('15 min')

# FILL MISSING VALUES FROM THE PREVIOUS WEEK'S VALUES
df_trend.fillna(df_trend.shift(-1,pd.Timedelta(weeks=1)),inplace=True)

# TRUNCATE THE PARTIAL ENTRIES FROM EITHER END
df_trend = egz.only_full_days(df_trend)

# IMPORT THE ADDITIONAL TRAINING FEATURES
df_extras = pd.read_csv('extras.csv',index_col=0,parse_dates=[0])

# CONSTRUCT THE MODEL
model = egz.MultiRFModel(
        data=df_trend,
        columns=['Main (kW)', 'Lighting (kW)', 'Plug load (kW)'],
        input_size=pd.Timedelta(weeks=4),
        gap_size=pd.Timedelta(days=1),
        output_size=pd.Timedelta(days=1),
        time_attrs=['dayofweek','dayofyear'],
        extra_features=df_extras,
        est_kwargs={'n_jobs':-1,
                     'n_estimators':128})
# TRAIN THE MODEL
model.train()

# MAKE THE NEXT PREDICTION
pred_vals, pred_stds = model.predict()

# EXPORT TO CSV FILES
pred_vals.to_csv('pred_vals.csv')
pred_stds.to_csv('pred_stds.csv')