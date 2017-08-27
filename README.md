This module contains methods and classes for trend data manipulation and prediction. It was designed with building electrical trend data in mind, but could easily be extended to other data such as natural gas or water usage.

# Installation
Download the `energize.py` file from the repo and put it in your working directory.
I tend to abbreviate the name as `egz` if making freqent references to its contents.

	import energize as egz

## Modeling
Most likely you will be using the `MultiRFModel` due to its ability to handle multiple columns of data at once. This is an extension of `SingleRFModel` which which performs the modeling for a single column of data. Usage of both is almost identical.

### Data
The data you pass in to a model will be a Pandas object (Series or Dataframe, depending on which model type).

The index should be at a consistent frequency (although it does not have to be explicitly defined in the object). The data should not have any missing entries. It also should be evenly divided into periods of the output size (e.g. if outputting a day of data, the trend data should start at a midnight time and end just before a midnight time).

If the source data has missing entries, it is suggested to deal with those before passing the data into the model. One approach is to call `fillna()` with a shifted copy of the data from the week prior. Another option is to `interpolate()` the data.

### Input/gap/output size
The input size controls how much of your data will be used as an input to predict future values. The output size controls how large the prediction period should be. The gap size controls the length of time between the end of the input (i.e. the current time) and the start of predictions, i.e. how far in advance predictions will be made.

These arguments should be a timedelta-like object such as `datetime.timedelta`, `pandas.Timedelta`, `numpy.timedelta64`, etc.

### Input downsampling
When the input data is at a high frequency it can be demanding to predict off every data point. The `sample_freq` parameter lets you pool together sections of the input at the desired pool size. The `sample_agg_method` determines how these pools will be aggregated. By default it takes the average value of each pool. The parameter takes a string representing one of the resampling methods (e.g. `'mean'`, `'sum'`, `'asfreq'`, etc.). See the [Pandas `resample()` documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html) for more options.

### Time features
In addition to historical trend data, features about the target prediction period can be used to make predictions. The `time_attrs` parameter can hold a list of attributes you would like to regress on stored as strings. These values can be [any of the time/date components](https://pandas.pydata.org/pandas-docs/stable/timeseries.html#time-date-components) of a `pandas.Timestamp` object.

### Extra features
If there are any additional variables that you want to make predictions with, these can be passed in to `extra_features` as a `pandas.DataFrame`. Each column can have an independent (but consistent) frequency. If there are multiple entries per output period then they will be treated as separate variable. That way if a property changes throughout the output period the model will be trained accordingly.

### Random Forest Keywords
The `est_kwargs` dictionary will pass arguments to the underlying `sklearn.ensembleRandomForestRegressor` estimator.

Here's two main ones to pay attention to:

 - `n_estimators` : number of trees in a forest. Higher is generally better.
      - default: 10
      - recommended: 50-100
 - `n_jobs` : number of cores to use for processing the forest
       - default: 1
       - recommended: -1 (use all cores)

### Column-specific features (only for MultiRFModel)
This functions the same way as the regular `extra_features` DataFrame. `column_features` is a dictionary of these types of DataFrames, where each key is a column title from the trend data and and the value is a feature table.

This is useful if only some columns are affected by certain variables.

### Training
Once a model has been contructed, (let's call it `model`), training can be performed with a simple call of `model.train()`

#### Under the hood
For a `SingleRFModel`, this process involves breaking up the trend data into windows (the combined size of the input, gap, and output sizes). These windows are then split into X and y components.

The X components are downsampled and aggregated as desired. The y components are used as target values. The y components also serve as references to pull sections from the extra feature tables and extract time attributes.

These segments of downsampled trend data, time features, and extra features are combined into training feature matrices. These matrices are used to call the `fit()` method of the underlying `sklearn.ensemble.RandomForestRegressor` objects.

A `MultiRFModel` simply trains each of its child `SingleRFModel` objects.

### Predicting
The period to be predicted will automatically be inferred from the historical trend data. The next available prediction period will automatically be predicted on.

For example, if your data is at 15 minute intervals, your gap size is 1 day, prediction size is 1 day and the final data point falls on Feb 2nd at 23:45:00, the prediction will span from 00:00:00 to 23:45:00 on Feb 4th.

Calling `model.predict()` will return a tuple of two objects. The first is a table of predicted values, the second is a table of corresponding prediction standard deviations.

#### Stats interpretation
The standard deviations are an estimate inspired by [this article](http://blog.datadive.net/prediction-intervals-for-random-forests/). A Random Forest is composed of a number of decision trees. This treats each set of trees as a sample of predictions of the overall forest. The table contains the standard deviations of those "samples".

Note that in the article I linked, intervals are selected by sample quantiles (they even make a note about this being more robust than assuming normality). In my case I used a normality assumption and returned a standard deviation value. The reason for this is so I can return a single measure of variance without any specific confidence level baked in to give the user more flexibility in their classification of an "anomaly"

To be conservative at smaller estimator counts, you may prefer to use a Student's t-distribution and take into account the number of tree estimators in your forest.

The `scipy `library makes it easy to work with these kinds of distributions. Distributions have a `ppf` method that will find a threshold Z-score from a desired one-tailed confidence interval level.

    z_thresh = scipy.stats.norm.ppf(0.95)
    thresh = pred_vals + z_thresh*pred_stds
    anomalies = true_vals[true_vals > thresh]
    est_diff = anomalies - thresh[anomalies.index]

From this example you could say with approximately 95% confidence that you could expect to save at least the contents of `est_diff`. If the trend values were power values you could estimate energy savings by using `egz.trapz` or if they were energy values you could just `sum()` the series.

### Exporting
You can call the `to_csv()` methods of the returned Pandas Series/DataFrame objects to save them as CSV files.

## Other module methods
Aside from the models there are some useful data manipulation functions included.

Some of the most useful ones:

 - `time_filter()` for filtering specific sections of data
 - `ical_ranges()` for using .ical files to pass to the filter
 - `intersect()` for extracting sections where tables overlap
 - `only_full_days()` for cutting off partial days
 - `trapz()` for estimating energy consumption from power use

More info available in the method signatures.

