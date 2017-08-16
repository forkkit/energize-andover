#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:05:54 2017

@author: matt
"""

import pandas as pd
import numpy as np
from icalendar import Calendar
import pytz
import datetime
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import optimize
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import math

"""
range_token_df: DataFrame, RangeToken --> DataFrame
Returns a dataframe filtered by the range token provided.

A RangeToken is either a datetime index (parial or formal)
or a tuple of start/end datetime indexes
"""
def range_token_df(data, token):
    if (type(token)==str):
        try:
            return data[token]
        except KeyError: #returns None
            print('[!] energize.py : range_token_df : ' + token+' not in range')
    else: # token is a start/end tuple
        return data[slice(*token)][:-1]

"""
data_in_range : DataFrame/Series, Data range --> DataFrame/Series
filters the input data by the date range provided
"""

def data_in_range(data, d_range):
    if (type(d_range)==list):
        return pd.concat(list(map(
                lambda token: range_token_df(data,token),
                d_range))).sort_index()
    else:
        return range_token_df(data,d_range)


"""
time_filter: DataFrame, ... --> DataFrame
filters data by properties like date and time

PARAMETERS:
data : DataFrame or Series with DateTimeIndex
*times: Tuple with start and end time strings as 'HH:MM'
	or list of such tuples
*include: Accepts a DataRange which is:
    1) A datetime index (partial or formal)
    2) A tuple of start and end datetime indexes (See 1)
        	Enter None to set to range min or max
    3) A list that contains any combination of types 1 and 2
*blacklist: range of dates to be excluded.
    See include parameter for acceptable format
    Overrides include parameter
*daysofweek: List of integers for days to be included
	0 = Mon, 6 = Sun
*months: List of integers for months to be included
    1 = Jan, 12 = Dec

starred parameters are optional
ranges are all inclusive
"""

def time_filter(data, **kwds):
    out = data
    if ('include' in kwds):
        out = data_in_range(out,kwds['include'])
    if ('times' in kwds):
        d_range = kwds['times']
        if type(d_range[0]) is tuple:
            out = pd.concat(list(map(
                    lambda subrange: out.between_time(*subrange),
                    d_range))).sort_index()
        else:
            out = out.between_time(*d_range)
    if ('daysofweek' in kwds):
        out = out[[day in kwds['daysofweek'] for day in out.index.weekday]]
    if ('months' in kwds):
        out = out[[month in kwds['months'] for month in out.index.month]]
    if ('blacklist' in kwds):
        out = out.drop(data_in_range(data, kwds['blacklist']).index, errors='ignore')
    return out

"""
convert_range_tz : DataRange(datetime.datetime), timezone --> DataRange
converts the ical default UTC timezone to the desired timezone
"""

def convert_range_tz(range_utc, local_tz):
    convert = lambda time: pytz.utc.localize(
            time.replace(tzinfo=None)).astimezone(
                    local_tz).replace(tzinfo=None)
    return tuple(map(convert,range_utc))

"""
ical_ranges: File Path --> ListOf DataRanges
reads the ics file at the given path, and turns the event start and end times
into data ranges that can be read by the time_filter function
"""
def ical_ranges(file):
    cal = Calendar.from_ical(open(file,'rb').read())
    ranges = []
    cal_tz = pytz.timezone(cal['X-WR-TIMEZONE'])
    for event in cal.subcomponents:
        event_range=(event['dtstart'].dt,event['dtend'].dt)
        if isinstance(event_range[0],datetime.datetime):
            event_range = convert_range_tz(event_range, cal_tz)
        ranges.append(event_range)
    return ranges


"""
mad: Data --> int
Get the median absolute deviation of the Series or each Dataframe column
"""

def mad(data, **kwds):
    return abs(data.sub(data.median(**kwds),axis=0)).median(**kwds)

"""
plot_normal: float, float --> void
Plot a normal distribution with the given mu and sigma
"""
def plot_normal(mu, sigma, **kwds):
    x = np.linspace(mu-8*sigma,mu+8*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma), **kwds)

"""
unstack_by_time: Series --> DataFrame
split timestamped series into date columns with common time index
"""
def unstack_by_time(series):
    stacked = series.copy()
    stacked.index = [stacked.index.time, stacked.index.date]
    return stacked.unstack()

"""
consecutives : Data, Offset --> GroupBy
organizes data in sections that are not more than the threshold time span apart
Group labels are just a count starting from 0

Example use:
    consecutives(df_energy, '15 min')
"""
def consecutives(data, threshold):
    dates = pd.Series(data.index, data.index)
    indicators = dates.diff() > pd.Timedelta(threshold)
    groups = indicators.apply(lambda x: 1 if x else 0).cumsum()
    return data.groupby(groups)

"""
energy_trapz : Data [opt: Offset ] --> int
uses a trapezoidal approximation to calculate energy used during the time period
Optional offset parameter determines how large of a time gap between entries
    is the threshold for data grouping
"""

def trapz(data, offset=None):
    if offset is None:
        offset = pd.Timedelta.max
    grouped = consecutives(data,offset)
    approx_kwh = lambda x: np.trapz(x,x.index).astype('timedelta64[h]').astype(int)
    return grouped.aggregate(approx_kwh).sum()

"""
lognorm_params: Series --> ( float, float, float )
Returns the shape, loc, and scale of the lognormal distribution of the sample data
"""

def lognorm_params(series):
    # resolve issues with taking the log of zero
    np.seterr(divide='ignore')
    log_data = np.log(series)
    np.seterr(divide='warn')
    log_data[np.isneginf(log_data)] = 0
    
    kde = stats.gaussian_kde(log_data)
    est_std = mad(log_data)*1.4826
    est_mu = optimize.minimize_scalar(lambda x: -1*kde.pdf(x)[0],
                                      method='bounded',
                                      bounds=(log_data.min(),log_data.max())).x
    return (est_std, 0, math.exp(est_mu))

"""
adjust_sample: Series *int --> Series
returns an adjusted version of the data that approximately follows the
energize fitted lognormal distribution

Buffer count (for setting the quantiles) defaults to 1 on each side (to take
the place of the 0th and 100th percentiles) and can optionally be changed
"""

def adjust_sample(series, buffer=1):
    fit_params = lognorm_params(series)
    s_sorted = series.sort_values()
    q_step = 1/(series.size+2*buffer-1)
    q_array = np.linspace(buffer*q_step, 1-buffer*q_step, series.size)
    quantiles=pd.Series(q_array, s_sorted.index).sort_index()
    return pd.Series(stats.lognorm.ppf(quantiles,*fit_params),
                     quantiles.index)
    
"""
intersect : Data Data --> (Data, Data)
returns a tuple of the data filtered by their common indexes
"""

def intersect(data1, data2):
    ixs = data1.index.intersection(data2.index)
    return(data1.loc[ixs],data2.loc[ixs])
    
    
def rolling_window(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def rolling_window2D(a,n):
    # a: 2D Input array 
    # n: Group/sliding window length
    return a[np.arange(a.shape[0]-n+1)[:,None] + np.arange(n)]

def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict([X[x]])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. , axis=0 ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2. , axis=0))
    return np.array(err_down), np.array(err_up)

def index_data(ixs_arr,data):
    return np.apply_along_axis(lambda ixs: data[ixs],0,ixs_arr)

class BaseModel:
    def __init__(self,data, td_input, td_gap, td_output,
                 sample_freq=pd.Timedelta(days=1), time_attrs=[],
                 extra_features=None,n_estimators=10):
        self.data = data
        self.n = len(data)
        self.data_freq = pd.Timedelta(data.index.freq)
        self.td_input = td_input
        self.td_gap = td_gap
        self.td_output = td_output
        self.sample_freq = sample_freq
        self.time_attrs = time_attrs
        self.n_estimators = n_estimators
        self.time_features = self._get_time_features()
        self.extra_features = (self._validated_feat(extra_features)
            if extra_features is not None else self._get_blank_feat())
        self.training_windows_ = self._get_training_windows()
        
    def _get_training_windows(self):
        input_size = int(self.td_input / self.data_freq)
        gap_size = int(self.td_gap / self.data_freq)
        output_size = int(self.td_output / self.data_freq)
        ix_windows = rolling_window(self.data.index,
                                    input_size + gap_size + output_size,
                                    output_size)
        X_ixs,_,y_ixs = np.split(ix_windows,[input_size,input_size+gap_size],1)
        return X_ixs,y_ixs
    
    def _get_time_features(self):
        df_f = self._get_blank_feat()
        for attr in self.time_attrs:
            df_f[attr] = getattr(df_f.index,attr)
        return df_f
    
    def _get_blank_feat(self):
        start = self.data.index.min()
        end = self.data.index.max() + self.td_gap + self.td_output
        ix = pd.date_range(start,end,freq=self.data_freq)
        return pd.DataFrame(index=ix)
    
    def _validated_feat(self,df):
        df_fallback = df.asfreq(self.td_output).ffill()
        return df.fillna(df_fallback)
        
    def _get_ixs(self,timestamp):
        X_ix = pd.date_range(timestamp - self.td_gap - self.td_input,
                              timestamp - self.td_gap,
                              freq=self.sample_freq,
                              closed='left')
        y_ix = pd.date_range(timestamp,
                             timestamp+self.td_output,
                             freq=self.data_freq,
                             closed='left')
        return X_ix, y_ix
        
    def _input_vector(self,series,timestamp):
        X_ix, y_ix = self._get_ixs(timestamp)
        data_feat = series[X_ix]
        time_feat = self.time_features.loc[timestamp]
        extra_feat = self.extra_features.loc[y_ix].stack().values
        vec = np.concatenate((data_feat, time_feat, extra_feat),
                             axis=0)
        return vec

    def _get_pred_std(self,rf,X):
        all_pred = np.array([t.predict(X) for t in rf])
        return np.std(all_pred,0).ravel()
    
    def _get_prediction(self,rf,series,timestamp):
        X_pred = [self._input_vector(series,timestamp)]
        pred_vals = rf.predict(X_pred)[0]
        pred_std = self._get_pred_std(rf, X_pred)
        _, y_ix = self._get_ixs(timestamp)
        pred = (pd.Series(pred_vals,y_ix),
                pd.Series(pred_std,y_ix))
        pred = list(map(lambda s: s.rename(series.name),pred))
        return pred
    
    def _get_pred_start_date(self,data):
        pred_start_date = (pd.Timestamp(data.index.date.max())
                            + timedelta(days=1) + self.td_gap)
        return pred_start_date
    
    
class SingleRFModel(BaseModel):
    def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)
            self.rf = RandomForestRegressor(n_estimators = self.n_estimators)
    
    def _get_training_arrays(self,series):
        X_ixs,y_ixs = self.training_windows_
        data_feat = np.array([series[w].asfreq(self.sample_freq)
                                for w in X_ixs])
        time_feat = np.array([self.time_features.loc[w] for w in y_ixs[:,0]])
        extra_feat = np.array([self.extra_features.loc[ixs].stack().values
                                for ixs in y_ixs])
                            
        X = np.concatenate((data_feat,
                            time_feat,
                            extra_feat),
            axis=1)
        y = np.array([series[w] for w in y_ixs])
        return X,y
    
    def train(self):
        self.rf.fit(*self._get_training_arrays(self.data))

    def predict(self):
        return self._get_prediction(self.rf,
                                    self.data,
                                    self._get_pred_start_date(self.data))

class MultiRFModel(SingleRFModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.columns = list(map(lambda col: self.data[col], self.data))
        self.estimators = list(map(lambda col:
            RandomForestRegressor(n_estimators=self.n_estimators),
            self.data))
            
    def train(self):
        for i,rf in enumerate(self.estimators):
            rf.fit(*self._get_training_arrays(self.columns[i]))
            
    def predict(self):
        pred_start_date = self._get_pred_start_date(self.data)
        preds = list(
                map(lambda rf,s: self._get_prediction(rf,s,pred_start_date),
                    self.estimators,self.columns))
        arr_vals, arr_std = list(zip(*preds))
        return (pd.concat(arr_vals,axis=1),
                pd.concat(arr_std,axis=1))

data_path = 'resources/2017 Mar - 2016 Aug - Electric - Detail - 24 Hrs.csv'

data_freq = '15 min'
pp_day = int(pd.Timedelta('1 day') / pd.Timedelta(data_freq))


df_energy = pd.read_csv(data_path, skipfooter=3, engine='python', index_col=0)
#df_energy.dropna(inplace=True)
df_energy.index = pd.to_datetime(df_energy.index)
df_energy = df_energy.dropna().resample(data_freq).asfreq()
df_energy = df_energy.groupby(df_energy.index.date).filter(lambda x: len(x)==pp_day)
df_energy.fillna(df_energy.shift(-pp_day*7),inplace=True)


no_school = ical_ranges('resources/no_school_2016-17.ics')
half_days = ical_ranges('resources/half_days_2016-17.ics')

df_school = time_filter(df_energy,
                            times=('07:40','14:20'),
                            include=('9/2/16','6/16/17'),
                            daysofweek=[0,1,2,3,4],
                            blacklist=no_school + half_days
                            + ['2/9/17','2/13/17','3/14/17','3/15/17'])

df_weekend = time_filter(df_energy,daysofweek=[5,6])
df_night = time_filter(df_energy,times=('23:00','04:00'),include=('2016-08',None))
df_summer = time_filter(df_energy,months=[7,8])

df_temp = pd.read_csv('resources/temperature.csv',
                      index_col=1,
                      na_values=-9999).drop('STATION',axis=1)

df_temp.index = pd.to_datetime(df_temp.index,format='%Y%m%d')