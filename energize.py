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
from scipy import optimize
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import multiprocessing as mp
import math
from collections import OrderedDict

"""
_range_token_df: DataFrame, RangeToken --> DataFrame
Returns a dataframe filtered by the range token provided.

A RangeToken is either a datetime index (parial or formal)
or a tuple of start/end datetime indexes
"""
def _range_token_df(data, token):
    if (type(token)==str):
        try:
            return data[token]
        except KeyError: #returns None
            print('[!] energize.py : range_token_df : ' + token+' not in range')
    else: # token is a start/end tuple
        return data[slice(*token)][:-1]

"""
_data_in_range : DataFrame/Series, Data range --> DataFrame/Series
filters the input data by the date range provided
"""

def _data_in_range(data, d_range):
    if (type(d_range)==list):
        return pd.concat(list(map(
                lambda token: _range_token_df(data,token),
                d_range))).sort_index()
    else:
        return _range_token_df(data,d_range)


"""
time_filter: DataFrame, ... --> DataFrame
filters data by properties like date and time

PARAMETERS:
data : DataFrame or Series with DateTimeIndex
*times: Tuple with start and end time strings as 'HH:MM'
	or list of such tuples
*include: Accepts a DataRange which is:
    1) A datetime reference (partial string or formal object)
    2) A tuple of start and end datetime references (See 1)
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
        out = _data_in_range(out,kwds['include'])
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
        out = out.drop(_data_in_range(data, kwds['blacklist']).index,
                       errors='ignore')
    return out

"""
_convert_range_tz : tuple(datetime,datetime), timezone --> DataRange
converts the ical default UTC timezone to the desired timezone
"""

def _convert_range_tz(range_utc, local_tz):
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
        if isinstance(event_range[0],datetime):
            event_range = _convert_range_tz(event_range, cal_tz)
        ranges.append(event_range)
    return ranges


"""
mad: Data --> int
Get the median absolute deviation of the Series or each Dataframe column
"""

def mad(data, **kwds):
    return abs(data.sub(data.median(**kwds),axis=0)).median(**kwds)

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

def inferred_freq(data):
    return pd.tseries.frequencies.to_offset(data.index.inferred_freq)

def only_full_days(data,freq=None):
    if freq is None:
        freq = inferred_freq(data)
    pp_day = pd.Timedelta(days=1) / freq
    return data.groupby(data.index.date).filter(lambda x: len(x)==pp_day)

class BaseModel:
    def __init__(self, data, input_size, gap_size, output_size,
                 sample_freq=pd.Timedelta(days=1), sample_agg_method='mean',
                 time_attrs=None, extra_features=None,est_kwargs=None):
        self.data = data.asfreq(data.index.inferred_freq)
        self.n = len(data)
        self.data_freq = pd.Timedelta(self.data.index.freq)
        self.input_size = input_size
        self.gap_size = gap_size
        self.output_size = output_size
        self.sample_freq = sample_freq
        self.sample_agg_method = sample_agg_method
        self.time_attrs = time_attrs if time_attrs is not None else []
        self.est_kwargs = est_kwargs if est_kwargs is not None else dict()
        self.training_windows_ = None
        self.extra_features = (extra_features if extra_features is not None
                               else self._get_blank_feat())

    def _get_training_windows(self):
        """Get the available input/output training windows for the data.
        These windows will be used to create slices of the feature tables,
        which will then form the X and y training arrays

        Returns
        -------
        X_ixs : list of [start, end] datetime64 pairs for the input values
        y_ixs : list of [start, end] datetime64 pairs for the output values
        """
        n_input = int(self.input_size / self.data_freq)
        n_gap = int(self.gap_size / self.data_freq)
        n_output = int(self.output_size / self.data_freq)
        ix_windows = rolling_window(self.data.index,
                                    n_input + n_gap + n_output,
                                    n_output)
        X_ixs,_,y_ixs = np.split(ix_windows,[n_input,n_input+n_gap],1)
        X_ixs = np.array(list(zip(X_ixs.min(1),X_ixs.max(1))))
        y_ixs = np.array(list(zip(y_ixs.min(1),y_ixs.max(1))))
        return X_ixs,y_ixs
    
    """ I might come back to this, the goal is to filter out windows that
    contain null values in the data or features.
    def _windows_where_valid(self,windows):
        no_nulls = lambda w: ~self.data[slice(*w)].isnull().values.any(),
        return list(filter(no_nulls, windows))"""
    
    def _get_blank_feat(self):
        """ Gets an empty data structure spanning the range of the model's data
        
        Returns
        -------
        blank_df : DataFrame
            A DataFrame with no columns and an index spanning from the
            start of the model's data to the end of the expected prediction
            period.
        """
        start = self.data.index.min()
        end = self.data.index.max() + self.gap_size + self.output_size
        ix = pd.date_range(start,end,freq=self.data_freq)
        return pd.DataFrame(index=ix)
    
    def _invalid_feat(self,data):
        return data.asfreq(self.output_size).isnull().values.any()
    
    def _validated_feat(self,feat):
        """ Ensures that the feature set is at a high enough frequency for the
        training windows (at least one point per feature per window)
        
        Parameters
        ----------
        feat : Series or DataFrame
            The set of features to validate
        
        Returns
        -------
        validated_feat : Series or DataFrame
            The validated feature table, forward filled where needed
        """
        
        #Ensure the extra features are at least the frequency of the output
        feat_freq = pd.Timedelta(inferred_freq(feat))
        if feat_freq > self.output_size:
            feat = feat.asfreq(self.output_size)
        fallback = feat.asfreq(self.output_size).ffill()
        return feat.fillna(fallback)
        
    def _get_ixs(self,timestamp):
        """ Gets the list of indexes for prediction of a date
        
        Parameters
        ----------
        timestamp : datetime
            The starting timestamp for the prediction
                
        Returns
        -------
        X_ix : DatetimeIndex
            The X indicies needed for historical value inputs
        y_ix : DatetimeIndex
            The y indicies of the prediction period
        """
        X_ix = pd.date_range(timestamp - self.gap_size - self.input_size,
                              timestamp - self.gap_size,
                              freq=self.sample_freq,
                              closed='left')
        y_ix = pd.date_range(timestamp,
                             timestamp+self.output_size,
                             freq=self.data_freq,
                             closed='left')
        return X_ix, y_ix
    
    def _get_windows(self,timestamp):
        X_w = (timestamp - self.gap_size - self.input_size,
               timestamp - self.gap_size - self.data_freq)
        y_w = (timestamp,
               timestamp + self.output_size - self.data_freq)
        return X_w,y_w
    
    def _get_time_feat(self, datetime):
        timestamp = pd.Timestamp(datetime)
        return [getattr(timestamp,attr) for attr in self.time_attrs]
    
    def _get_pred_start_date(self,data):
        """ Gets the inferred starting date for the next prediction based off
        the existing data index and the desired gap size.
        
        Parameters
        ----------
        data : Series or DataFrame
            The historical data
        
        Returns
        -------
        pred_start_date : datetime
            The inferred starting date of the next prediction
        """
        pred_start_date = (pd.Timestamp(data.index.max())
                            + self.data_freq + self.gap_size)
        return pred_start_date
    
    def to_string(val):
        """ Extracts some descriptive information from various types of data.
        These are spefically tested for the use of creating logs from different
        types of model data. For instance, it will return a formatted list
        of column headers from a DataFrame, or the title of a Series.
        
        Parameters
        ----------
        val : object
            An arbitrary value
            
        Returns:
        out : string
            The formatted output string
        """
        if isinstance(val,str):
            out = val
        elif val is None:
            out = ''
        elif isinstance(val,dict):
            out = ', '.join('{}:{}'.format(key,BaseModel.to_string(val)) for key,val in val.items())
        elif isinstance(val,(list,pd.DataFrame)):
            out = ', '.join(val)
        elif isinstance(val,pd.Series):
            out = val.name
        else:
            out = str(val)
        return out

    def log(self):
        """ Creates a table of pertinent information for the current state of
        a model, used for making logs of predictions
        
        Returns
        -------
        log : Series
            The index contains class attribute names and other data labels,
            the rows contain the corresponding data formatted as a string
        """
        parse = BaseModel.to_string
        log_attrs = ['n','data_freq','input_size','gap_size','output_size',
                      'sample_freq','sample_agg_method','time_attrs',
                      'extra_features','est_kwargs']
        
        log = pd.Series()
        log['timestamp'] = datetime.strftime(datetime.now(),
                                             '%Y-%m-%d %H:%M:%S')
        log['pred_start'] = parse(self._get_pred_start_date(self.data))
        log['pred_end'] = parse(self._get_pred_start_date(self.data)
                                    + self.output_size - self.data_freq)
        log['data_start'] = parse(self.data.index.min())
        log['data_end'] = parse(self.data.index.max())
        for attr in log_attrs:
            log[attr] = parse(getattr(self,attr))
        return log

class SingleRFModel(BaseModel):
    """ A Random Forest Model for forecasting a single set of data
    (i.e. a single circuit panel)
    
    Parameters
    ----------
    data : Series
        The historical data values, and the data type to be forecasted. There
        should not be any missing values in the data - that should be handled
        before object initialization. That data should also be at a set
        frequency (if it is not explicity store in `pandas.DatetimeIndex.freq
        it will be inferred)
        
    input_size : timedelta
        The size of the input period used for selecting historical data values
        
    gap_size : timedelta
        The size of the gap between the last of the historical data and the
        start of the output period
        
    output_size : timedelta
        The size of the output period (to be forecasted)
        
    sample_freq : timedelta, optional (default=timedelta(days=1))
        The chunk size that will break up the historical data and aggregate it
        into features (by mean). By default it will look at the daily averages
        among the input periods.
        
    sample_agg_method : string, optional (defaul='mean')
        A string representing the aggregation method to be called on the
        resampled historical data. For example, with the default value of
        'mean' it will pull data features from
        data.resample(sample_freq).mean() for the purpose of decreasing feature
        load on the estimator.
    
    time_attrs : list of strings, optional (default=None)
        The list of DatetimeIndex attributes to use as regression features.
        For example, setting it to `['month','dayofweek'] will consider the
        month and day of the week of only the first output element
        
    extra_features : DataFrame, optional (default=None)
        Table of additional features to regress on. If the frequency is wider
        than the output period, it will be foreward filled as needed to
        maintain consistent input vector shape. If the frequency is more narrow
        than the output period, all the entries that lie within an output
        will be used as independent features.
        
        For example, if one column is average temperature at weekly frequency,
        the second column is occupancy data at hourly frequency, and the output
        period is one day, then the temperature column will be foreward filled
        to have one point per day and 24 occupancy points will be used as 
        features.
        
    est_kwargs : dict, optional (default=None)
        Additional keyword arguments to be passed to the
        sklearn.ensemble.RandomForestRegressor estimator
        
    """
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.rf = RandomForestRegressor(**self.est_kwargs)
    
    def _aggregated_data_features(self):
        """ Aggregates the data features. Since it may be resource
        intensive to use every data point as a feature, this lets you combine
        a group of data points (as specified by `BaseModel.sample_freq`) into
        their mean value.
                
        Returns
        -------
        agg : Series
            The model data, downsampled and aggregated by mean
        """
        resampler = getattr(self.data.resample(self.sample_freq),
                            self.sample_agg_method)
        return resampler()
    
    #@time_func
    def _get_feats(self, X_windows, y_windows):
        agg_data = self._aggregated_data_features()
        data_feat = np.array([agg_data[slice(*w)]
                                for w in X_windows])
        time_feat = np.array([self._get_time_feat(d)
                                for d in np.array(y_windows)[:,0]])
        extra_feat = np.array([self.extra_features[slice(*w)].stack().values
                                for w in y_windows])
        return (data_feat,time_feat,extra_feat)

    def _get_training_arrays(self):
        X_windows,y_windows = self.training_windows_
        feature_arrs = self._get_feats(*self.training_windows_)
        X = np.concatenate(feature_arrs, axis=1)
        y = np.array([self.data[slice(*w)] for w in y_windows])
        return X,y
        
    def _input_vector(self,timestamp):
        """ Get the input vector for a prediction.
        
        Parameters
        ----------
        timestamp : datetime
            The starting timestamp for the prediction
            
        Returns
        -------
        vec : ndarray
            The input vector of shape (n_features)
        """
        X_w,y_w = self._get_windows(timestamp)
        vec = np.hstack(self._get_feats([X_w],[y_w])).flatten()
        return vec

    def _get_pred_std(self,X):
        """ Get the standard deviations among a forests' decision trees for
        a certain prediction
        
        Parameters
        ----------
        X : ndarray
            An input vector
            
        Returns
        -------
        std : ndarray
            A list of standard deviations for each of the output variables
        """
        all_pred = np.array([t.predict(X) for t in self.rf])
        return np.std(all_pred,0).ravel()
    
    def _get_prediction(self,timestamp):
        """ Get the predicted values and standard deviations starting at a
        certain time
        
        Parameters
        ----------
        timestamp : datetime
            The starting time of the prediction
            
        Returns
        vals : Series
            The predicted values
        std : Series
            The prediction standard deviations
        """
        
        X_pred = [self._input_vector(timestamp)]
        pred_vals = self.rf.predict(X_pred)[0]
        pred_std = self._get_pred_std(X_pred)
        _, y_ix = self._get_ixs(timestamp)
        vals,std = (pd.Series(pred_vals,y_ix),
                    pd.Series(pred_std,y_ix))
        vals,std = [s.rename(self.data.name) for s in (vals,std)]
        return vals,std
    
    #@time_func
    def train(self):
        if self.training_windows_ is None:
            self.training_windows_ = self._get_training_windows()
        if self._invalid_feat(self.extra_features):
            self.extra_features = self._validated_feat(self.extra_features)
        n_samples = len(self.training_windows_[0])
        sample_weight=1-np.logspace(np.log10(0.5),-3,n_samples)+1e-3
        self.rf.fit(*self._get_training_arrays(),
                    sample_weight=sample_weight)

    def predict(self):
        """ Predict the next period of data
        
        The predicted period will be inferred based off the last entry of
        the historical data, gap size, and output size.
        
        For example, if the historical data contains up to the end of 1/1/17,
        the gap size is 1 day, and the output size is 1 week, the predicted
        period will span from the start of 1/3/17 to the end of 1/9/17
        
        Returns
        -------
        vals : Series
            List of predicted values for each of the data columns.
        std : Series
            List of output standard deviations. This is calculated from the
            standard deviation of the forests' decision trees for each output
            variable and is merely an estimation of the true prediction
            variance. Accuracy is increased with a higher estimator count.
        """
        return self._get_prediction(self._get_pred_start_date(self.data))
    
    def reload_data(self, data=None, extra_features=None):
        """ Reload data sources to allow continued predictions from an
        existing model. Not intended to be used extensively, preferred to
        remake/train a model with the new data.
        
        NOTE:
        Updated attributes must have the same format as the existing attributes
        or prediction will fail
        
        Parameters
        ----------
        data : Series, optional (default=None)
            Updated set of historical data
        extra_features : DataFrame, optional (default=None)
            Updated table of additional features.
        """
        if data is not None:
            self.data = data
        if extra_features is not None:
            self.extra_features = extra_features


class MultiRFModel(BaseModel):
    """ A Random Forest regression model for forecasting multiple sets of data
    at once. It creates multiple child SingleRFModel objects but avoids
    repitition of processes like training window calculations
    
    Parameters
    ----------
    *args, *kwargs :
        Arguments to be passed to the children `SingleRFModel` objects (see
        `SingleRFModel` documentation for details)
        
    data : DataFrame
        Table of data sets to be regressed and forecasted of shape
        (n_samples, s_sets). There should not be any missing values in the
        data - that should be handled before object initialization. That data
        should also be at a set frequency (if it is not explicity store in
        `pandas.DatetimeIndex.freq it will be inferred)
    
    columns : list of strings, optional (default=None)
        List of which columns you want to model from the data table
    
    column_features : dict of DataFrames, optional (default=None)
        Table of column-specific extra features to be regressed on. See
        `SingleRFModel.extra_features` for details of the required format of
        the DataFrames. The dict keys should correspond to column names in
        `MultiRFModel.data`
        
    """

    def __init__(self, data, *args, columns=None, column_features=None, **kwargs):
        super().__init__(data,*args,**kwargs)
        self.columns = columns if columns is not None else list(data.columns)
        self.models = OrderedDict(
                [(col,SingleRFModel(data[col],*args,**kwargs))
                for col in self.columns])
        self.column_features = column_features
        if column_features is not None:
            self._add_column_features(column_features)
    
    def _add_column_features(self,column_features):
        common_feat = self.extra_features
        for col,col_feat in column_features.items():
            self.models[col].extra_features = pd.concat(
                    (common_feat,col_feat),axis=1)
    
    def subtrain(item):
        """ Used for training submodels. This is a helper function needed for
        the multiprocessing mapper in the `train` method.
        
        Parameters
        ----------
        item : tuple of (string, SingleRFModel)
            A key, value pair from the `MultiRFModel.models` dictionary.
            
        Returns
        -------
        col : string
            The column name
        trained : SingleRFModel
            A trained copy of the inputted model
        """
        col, model = item
        model.train()
        return (col,model)
    
    #@time_func
    def train(self):
        """ Train the model to enable predictions. This trains each of the
        child `SingleRFModel` objects
        """
        self.training_windows_ = self._get_training_windows()
        for model in self.models.values():
            model.training_windows_ = self.training_windows_
        pool= mp.Pool(processes=mp.cpu_count())
        results = pool.map(MultiRFModel.subtrain,self.models.items())
        pool.close()
        pool.join()
        self.models = OrderedDict(results)
        
    def predict(self):
        """ Predict the next period of data
        
        The predicted period will be inferred based off the last entry of
        the historical data, gap size, and output size.
        
        For example, if the historical data contains up to the end of 1/1/17,
        the gap size is 1 day, and the output size is 1 week, the predicted
        period will span from the start of 1/3/17 to the end of 1/9/17
        
        Returns
        -------
        vals : DataFrame
            Table of predicted values for each of the data columns.
        std : DataFrame
            Table of output standard deviations. This is calculated from the
            standard deviation of the forests' decision trees for each output
            variable and is merely an estimation of the true prediction
            variance. Accuracy is increased with a higher estimator count.
        """
        preds = [model.predict() for model in self.models.values()]
        arr_vals, arr_std = list(zip(*preds))
        return (pd.concat(arr_vals,axis=1),
                pd.concat(arr_std,axis=1))
        
    def reload_data(self, data=None, extra_features=None,
                    column_features=None):
        """ Reload data sources to allow continued predictions from an
        existing model. Not intended to be used extensively, preferred to
        remake/train a model with the new data.
        
        NOTE:
        Updated attributes must have the same format as the existing attributes
        or prediction will fail
        
        Parameters
        ----------
        data : DataFrame, optional (default=None)
            Updated table of historical data
        extra_features : DataFrame, optional (default=None)
            Updated table of additional common features.
        column_features : dict, optional (default=None)
            Updated dictionary with column names as keys and feature DataFrames
            as values.
        """
        for col,model in self.models.items():
            model.reload_data(data[col],extra_features)
        if column_features is not None:
            self._add_column_features(column_features)
            
    def log(self):
        unique_attrs = ['column_features']
        log = BaseModel.log(self)
        for attr in unique_attrs:
            log[attr] = BaseModel.to_string(getattr(self,attr))
        return log