#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:16:55 2017

@author: matt
"""
import numpy as np
import scipy.stats as stats
import pandas as pd
import energize as egz
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

data_path = 'resources/2017 Mar - 2016 Aug - Electric - Detail - 24 Hrs.csv'

df_energy = pd.read_csv(data_path, skipfooter=3, engine='python', index_col=0)
df_energy.dropna(inplace=True)
df_energy.index = pd.to_datetime(df_energy.index)

no_school = egz.ical_ranges('resources/no_school_2016-17.ics')
half_days = egz.ical_ranges('resources/half_days_2016-17.ics')

df_school = egz.time_filter(df_energy,
                            times=('07:40','14:20'),
                            include=('9/2/16','6/16/17'),
                            daysofweek=[0,1,2,3,4],
                            blacklist=no_school + half_days
                            + ['2/9/17','2/13/17','3/14/17','3/15/17'])

df_night = egz.time_filter(df_energy,times=('23:00','04:00'))
school_main = df_school['Main (kW)']+0.00001
"""
x = np.linspace(200,700,100)
params = stats.lognorm.fit(school_main)
arg = params[:-2]
loc = params[-2]
scale = params[-1]
pdf = stats.lognorm.pdf(x, loc=loc, scale=scale, *arg)
"""

log_school_main = np.log(school_main)
x = pd.Series(np.linspace(log_school_main.min(), log_school_main.max(), 10000))



kde = stats.gaussian_kde(log_school_main)
kde_pdf = pd.Series(kde.pdf(x))

est_mu = x[kde_pdf[kde_pdf == kde_pdf.max()].index].iloc[0]
est_sd = egz.mad(log_school_main)*1.4826

"""
params = stats.norm.fit(log_school_main)
est_mu = params[0]
est_sd = params[1]
"""
"""
est_mu = log_school_main.median()
est_sd = egz.mad(log_school_main)*1.4826
"""

norm_pdf = stats.norm.pdf(x, est_mu, est_sd)

    

max_y = kde_pdf.max()

np.log(school_main).plot.kde(figsize=(10,5), xlim=(log_school_main.min(),log_school_main.max())).set(
        title='Log of Main Power Using Estimated Population Parameters\n'+
                'Maximum of Sample KDE and Scaled MAD')
#plt.axhline(max_y,color='g', linestyle='--')
#plt.axvline(stats.norm.ppf(0.95, est_mu, est_sd),color='r',linestyle='--')

blue_patch = mpatches.Patch(color='tab:blue', label='Sample KDE')
yellow_patch = mpatches.Patch(color='tab:orange', label='N(max(KDE), MAD*1.4826)')

plt.legend(handles=[blue_patch, yellow_patch])


plt.plot(x,norm_pdf)
