'''
Machine Learning Pipeline

Ben Fogarty

2 May 2019
'''

import dateutil.relativedelta as relativedelta
from textwrap import wrap
from sklearn import *
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def read_csv(filepath, cols=None, col_types=None, index_col=None):
    '''
    Imports a CSV file into a pandas data frame, optionally specifying columns
    to import, the types of the columns, and an index column.

    Inputs:
    filepath (str): the path to the file
    cols (list of strings): an optional list of columns from the data to import;
        can only be used if the first line of the csv is a header row
    col_types (dict mapping strings to types): an optional dictionary specifying
        the data types for each column; each key must be a the name of a column
        in the dataset and the associated value must be a pandas datatype (valid
        types listed here: http://pandas.pydata.org/pandas-docs/stable/
        getting_started/basics.html#dtypes)
    index_col (str or list of strs): an optional column name or list of column
        names to index the rows of the dataframe with

    Returns: pandas dataframe
    '''
    return pd.read_csv(filepath, usecols=cols, dtype=col_types,
                       index_col=index_col)

def count_per_categorical(df, cat_column):
    '''
    Summaries the number of observations associated with each value in a given
    categorical column and shows the distribtuion of observations across
    categories.
    
    Inputs:
    df (pandas dataframe): the dataset
    cat_column (str): the name of the categorical column

    Returns: tuple of pandas dataframe, matplotlib figure
    '''
    count_per = df.groupby(cat_column)\
                  .count()\
                  .iloc[:, 0]\
                  .rename('obs_per_{}'.format(cat_column))

    summary = count_per.describe()
    fig = show_distribution(count_per)

    return summary, fig

def show_distribution(series):
    '''
    Graphs a histogram and the box plot of numeric type series and a bar plot
    of categorial type series.

    Inputs:
    df (pandas series): the variable to show the distribution of

    Returns: matplotlib figure

    Citations:
    Locating is_numeric_dtype: https://stackoverflow.com/questions/19900202/
    '''
    series = series.dropna()
    sns.set()
    if pd.api.types.is_numeric_dtype(series):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        sns.distplot(series, kde=False, ax=ax1)
        sns.boxplot(x=series, ax=ax2, orient='h')
        ax1.set_title('Histogram')
        ax1.set_ylabel('Count')
        ax1.set_xlabel('')
        ax2.set_title('Box plot')
    else:
        f, ax = plt.subplots(1, 1)
        val_counts = series.value_counts()
        sns.barplot(x=val_counts.index, y=val_counts.values, ax=ax)
        ax.set_ylabel('Count')

    f.suptitle('Distribution of {}'.format(series.name))
    f.subplots_adjust(hspace=.5, wspace=.5)

    return f

def pw_correlate(df, variables=None, visualize=False):
    '''
    Calculates a table of pairwise correlations between numeric variables.

    Inputs:
    df (pandas dataframe): dataframe containing the variables to calculate
        pairwise correlation between
    variables (list of strs): optional list of variables to calculate pairwise
        correlations between; each passed str must be name of a numeric type
        (including booleans) column in the dataframe; default is all numeric
        type variables in the dataframe
    visualize (bool): optional parameter, if enabled the function generates
        a heat map to help draw attention to larger correlation coefficients

    Returns: pandas dataframe

    Wrapping long axis labels: https://stackoverflow.com/questions/15740682/
                               https://stackoverflow.com/questions/11244514/
    '''
    if not variables:
        variables = [col for col in df.columns
                         if pd.api.types.is_numeric_dtype(df[col])]

    corr_table = np.corrcoef(df[variables].dropna(), rowvar=False)
    corr_table = pd.DataFrame(corr_table, index=variables, columns=variables)

    if visualize:
        sns.set()
        f, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_table, annot=True, annot_kws={"size": 'small'},
                    fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, square=True,
                    cmap='coolwarm', ax=ax)

        labels = ['-\n'.join(wrap(l.get_text(), 16)) for l in ax.get_yticklabels()]
        ax.set_yticklabels(labels)
        labels = ['-\n'.join(wrap(l.get_text(), 16)) for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', rotation=0, labelsize='small')
        ax.tick_params(axis='x', rotation=90, labelsize='small')

        ax.set_title('Correlation Table')
        f.tight_layout()
        f.show()

    return corr_table

def summarize_data(df, grouping_vars=None, agg_cols=None):
    '''
    Groups rows based on the set of grouping variables and report summary
    statistics over the other numeric variables.

    Inputs:
    df (pandas dataframe): dataframe containing the variables to calculate
        pairwise correlation between
    grouping_vars (list of strs): optional list of variables
        to group on before aggregating; each passed str must be name of a column
        in the dataframe; if not included, no grouping is performed
    agg_cols (list of strs): optional list of variables to
        aggregate after grouping; each passed str must be name of a column in
        the dataframe; default is all numeric type variables in the dataframe

    Returns: pandas dataframe
    '''
    if not grouping_vars:
        grouping_vars = []
    if agg_cols:
        keep =  grouping_vars + agg_cols
        df = df[keep]

    if grouping_vars:
        summary = df.groupby(grouping_vars)\
                    .describe()
    else:
        summary = df.describe()

    return summary.transpose()

def find_ouliers_univariate(series):
    '''
    Identifies values in a series that fall more than 1.5 * IQR below the first
    quartile or 1.5 * IQR above the third quartile.

    Inputs:
    series (pandas series): the series to look for outliers in, must be numeric

    Returns: pandas series
    '''
    quartiles = np.quantile(series.dropna(), [0.25, 0.75])
    iqr = quartiles[1] - quartiles[0]
    lower_bound = quartiles[0] - 1.5 * iqr
    upper_bound = quartiles[1] + 1.5 * iqr

    return (lower_bound > series) | (upper_bound < series)

def find_outliers(df, excluded=None):
    '''
    Identifies outliers for each numeric column in a dataframe, and returns a
    dataframe matching each record with the columns for which it is an outlier
    and the number and percent of checked columns for which a is an outlier.
    Outlier is defined as any value thats fall more than 1.5 * IQR below the
    first quartile or 1.5 * IQR above the third quartile of all the values in a
    column.

    Inputs:
    df (pandas dataframe): the dataframe to find outliers in
    excluded (str list of strs): optional column name or a list of columns names
        not to look for outliers in; default is including all numeric columns

    Returns: pandas series
    '''
    if not excluded:
        excluded = []

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    outliers = df[numeric_cols]\
                 .drop(excluded, axis=1, errors='ignore')\
                 .apply(find_ouliers_univariate, axis=0)
    outliers['Count Outlier'] = outliers.sum(axis=1, numeric_only=True)
    outliers['% Outlier'] = (outliers['Count Outlier'] /
                             (len(outliers.columns) - 1) * 100)

    return outliers

def replace_missing(series):
    '''
    Replaces missing values in a series with the median value if the series is
    numeric and with the modal value if the series is non-numeric.

    Inputs:
    series (pandas series): the series to replace data in

    Returns (pandas series)
    '''
    if pd.api.types.is_numeric_dtype(series):
        median = np.median(series.dropna())
        return series.fillna(median)
    else:
        mode = series.mode().iloc[0]
        return series.fillna(mode)

def preprocess_data(df):
    '''
    Removes missing values, replacing them with the median value in numeric
    columns and the modal value in non-numeric columns.

    Inputs:
    df (pandas dataframe): contains the data to preprocess

    Returns: pandas dataframe
    '''
    return df.apply(replace_missing, axis=0)

def cut_variable(series, bins, labels=None, kwargs=None):
    '''
    Discretizes a continuous variable into bins. Bins are half-closed, [a, b).

    Inputs:
    series (pandas series): the variable to discretize
    bins (int or list of numerics): the binning rule:
        - if int: cuts the variable into approximate n bins with an approximately
          equal number of observations (there may be fewer bins or some bins
          with a substantially larger number of observations depending on the
          distribution of the data)
        - if list of numerics: cuts the variable into bins with edges determined
          by the sorted numerics in the list; any values not covered by the
          specified will be labeled as missing
    labels (list of str): optional list of labels for the bins; must be the same
        length as the number of bins specified
    kwargs (dictionary): keyword arguments to pass to either pd.cut or pd.qcut

    Return: pandas series
    '''
    if not kwargs:
        kwargs = {}

    if isinstance(bins, int):
        return pd.qcut(series, bins, labels=labels, duplicates='drop', **kwargs)\
                 .astype('category')

    return pd.cut(series, bins, labels=labels, include_lowest=True, **kwargs)\
             .astype('category')

def create_dummies(df, columns, kwargs=None):
    '''
    Transforms variables into a set of dummy variables.

    Inputs:
    df (pandas dataframe/series): the data to transform dummies in;
        all columns not being converted to dummies must be numeric
        types
    columns (list of strs): column names containing categorical
        variables to convert to dummy variables
    kwargs (dict): optional keyword arguments to pass to pd.get_dummies

    Returns: pandas dataframe where the columns to be converted is replaced with
        columns containing dummy variables
    '''
    if not kwargs:
        kwargs = {}

    return pd.get_dummies(df, columns=columns, **kwargs)

def cut_frequency(series, value=None, quantile=None):
    '''
    Transforms variable into true/false based on whether how frequently its
    values appear in the dataset.

    Inputs:
    series (pandas series): the variable to transform
    value (int or float): the threshold value; observations with values of the
        variable that appear at least this many times will be evaluate as true
        and observations that appear fewer than this number of times will be
        evaluated as false; one of value or quantile must be specified
    quantile (float) the threshold value; observations with values of the
        variable that appear at least at or above this quantile evaluate as true
        and observations that appear fewer than this quantil will be
        evaluated as false; one of value or quantile must be specified

    Returns: pandas series:
    '''
    assert bool(value) + bool(quantile) == 1, ("Exactly one of value or " +
                                               "quantile must be specified.")
    val_counts = series.value_counts()
    if quantile:
        value = np.quantile(val_counts, quantile)

    return series.apply(lambda x: val_counts[x] >= value)

def create_time_diff(start_dates, end_dates):
    '''
    Calculates the time difference between two date columns.

    Inputs: 
    start_dates (pandas series): the start dates to calculate the difference
        from; column should be have type datetime
    end_dates (pandas series): the end dates to calculate the difference to;
        columns should have type datetime

    Returns: pandas series of timedelta objects
    '''
    return end_dates - start_dates

def report_n_missing(df):
    '''
    Reports the percent of missing (float.NaN or None) values for each column
    in a dataframe.

    Inputs:
    df (pandas dataframe): the dataset

    Returns: pandas dataframe
    '''
    missing = pd.DataFrame(columns=['# Missing'])

    for column in df.columns:
        missing.loc[column, '# Missing'] = np.sum(df[column].isna())

    missing['% Missing'] = missing['# Missing'] / len(df) * 100

    return missing

def visualize_decision_tree(dt, feature_names, class_names, filepath='tree'):
    '''
    Saves and opens a PDF visualizing the specified decision tree.

    Inputs:
    dt (sklearn.tree.DecisionTreeClassifier): a trained decision tree classifier
    feature_names (list of strs): a list of the features the data was trained
        with; must be in the same order as the features in the dataset
    class_names (list of strs): a list of the classes of the target attribute
        the model is predicting; must match the target attribute values if
        those values were given in ascending order
    filepath (str): optitional parameter specifying the output path for the
        visualization (do not include the file extension); default is 'tree' in
        the present working directory

    Citations:
    Guide to sklearn decision trees: https://scikit-learn.org/stable/modules/
        tree.html
    sklearn.tree.export_graphviz docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.export_graphviz.htm
    '''
    dot_data = tree.export_graphviz(dt, None, feature_names=feature_names,
                                    class_names=class_names, filled=True)
    graph = graphviz.Source(dot_data)
    output_path = graph.render(filename=filepath, view=True)

def generate_classifiers(features, target, models):
    '''
    Generates one or more classifiers to predict a target attribute (target)
    based on other attributes (features).

    Inputs:
    features (pandas dataframe): Data for features to build the classifier(s)
        with; all columns must be numeric in type
    target (pandas series): Data for target attribute to build the classifier(s)
        with; should be categorical data in a numerical form
    models (list of dicts): A list of dictionaries specifying the classifier
        models to generate. Each dictionary must contain a "model" key with a
        value specifying the type of model to generate; currently supported
        types are listed below. All other entries in the dictionary are optional
        and should have the key as the name of a parameter for the specified
        classifier and the value as the desired value of that parameter.

    Returns: list of trained classifier objects

    Currently supported model types:
    'dt': sklearn.tree.DecisionTreeClassifier
    'lr': sklearn.linear_model.LogisticRegression
    'knn': sklearn.neighbors.KNeighborsClassifier
    'svc': sklearn.svm.LinearSVC
    'rf': sklearn.ensemble.RandomForestClassifier
    'boosting': sklearn.ensemble.AdaBoostClassifier
    'bagging': sklearn.ensemble.BaggingClassifier

    Example usage:
    generate_classifiers(x, y, [{'model': 'dt', 'max_depth': 5}, 
                                {'model': 'knn', 'n_neighbors': 10}])

    The above line will generate two classifiers, the first of which is a 
    decision tree classifier with a max depth of 5, and the second of which is a
    k nearest neighbors model with k=10.

    For more information on valid parameters to include in the dictionaries,
    consult the sklearn documentation for each model.
    '''
    model_class = {'dt': tree.DecisionTreeClassifier,
                   'lr': linear_model.LogisticRegression,
                   'knn': neighbors.KNeighborsClassifier,
                   'svc': svm.LinearSVC,
                   'rf': ensemble.RandomForestClassifier,
                   'boosting': ensemble.AdaBoostClassifier,
                   'bagging': ensemble.BaggingClassifier}

    classifiers = []

    for model_specs in models:
        model_type = model_specs.pop('model')
        model = model_class[model_type](**model_specs)
        model.fit(features, target)
        classifiers.append(model)

    return classifiers

def evaluate_classifier(model, features, target, thresholds):
    '''
    Calculates a number of evaluation metrics (accuracy precision, recall, and
    F1 at different levels and AUC-ROC) and generates a graph of the
    precision-recall curve for a given model.

    models (trained sklearn classifier): the model to evaluate
    features (pandas dataframe): test data for the features in the
        classifier; structure of the data (columns and column types) must
        match the data used to train the classifier
    target (pandas series): test data for the target attribute the classifier
        predicts (i.e. the true class of each observation in the test data)
    thresholds (list of ints): a list of different threshold levels to use when
        calculating precision, recall and F1
    
    Returns: tuple of pandas series and matplotlib figure
    '''
    index = [['Accuracy'] * len(thresholds) +['Precision'] * len(thresholds) + 
             ['Recall'] * len(thresholds) + ['F1'] * len(thresholds),
             thresholds * 4]
    index = list(zip(*index))
    index.append(('AUC-ROC', None))
    index = pd.MultiIndex.from_tuples(index, names=['Metric', 'Threshold']) 
    evaluations = pd.Series(index=index)

    pred_prob = model.predict_proba(features)[:, 1]

    for threshold in thresholds:
        pred_class = pred_prob > threshold
        evaluations['Accuracy', threshold] = metrics.accuracy_score(target, pred_class)
        evaluations['Precision', threshold] = metrics.precision_score(target, pred_class)
        evaluations['Recall', threshold] = metrics.recall_score(target, pred_class)
        evaluations['F1', threshold] = metrics.f1_score(target, pred_class)

    evaluations['AUC-ROC', None] = metrics.roc_auc_score(target, pred_prob)

    sns.set()
    fig, ax = plt.subplots()
    precision, recall, thresholds = metrics.precision_recall_curve(target, pred_prob)
    sns.lineplot(recall, precision, drawstyle='steps-post', ax=ax)

    ax.fill_between(recall, precision, alpha=0.2, step='pre')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    fig.suptitle('Precision-Recall Curve')


    return evaluations, fig

def create_temporal_splits(df, date_col, time_length, start_date=None): 
    '''
    Splits into different sets by time intervals.

    Inputs:
    df (pandas dataframe): the full dataset to split
    date_col (str): the name of the column in the dataframe containing the date
        attribute to split on
    time_length (dictionary): specifies the time length of each split, with
        strings of units of time (i.e. hours, days, months, years, etc.) as keys
        and integer as values; for example 6 months would be {'months': 6}
    start_date (str): the first date to include in the splits; value should be
        in the form "yyyy-mm-dd"

    Returns: list of pandas dataframes, the split datasets in temporal order
    '''
    time_length = relativedelta.relativedelta(**time_length)
    if start_date:
        start_date = pd.to_datetime(start_date, format='yyyy-mm-dd')
        df = df[df[date_col] > start_date]
    else:
        start_date = min(df[date_col])
    
    splits = []
    max_date = max(df[date_col])
    i = 0
    while start_date + (i * time_length) < max_date:
        lower_mask = (start_date + (i * time_length)) <= df[date_col]
        upper_mask = df[date_col] < (start_date + ((i + 1) * time_length))
        splits.append(df[lower_mask & upper_mask])
        i += 1

    return splits
