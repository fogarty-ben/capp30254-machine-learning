'''
Simple Machine Learning Pipeline

Ben Fogarty

18 April 2018
'''

import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from textwrap import wrap
from sklearn import tree

def read_csv(filepath, cols=None, col_types=None):
    '''
    Imports a CSV file into a pandas data frame, optionally specifying columns
    to import and the types of the columns.

    Inputs:
    filepath (str): the path to the file
    row_index (str or list of strs): an optional column name or list of column
        names to index the rows of the dataframe with
    cols (list of strings): an optional list of columns from the data to import;
        only to be used if the first line of the csv is a header row
    col_types (dict mapping strings to types): an optional dictionary specifying
        the data types for each column; each key must be a the name of a column
        in the dataset and the associate value must be a pandas datatype (valid
        types listed here: http://pandas.pydata.org/pandas-docs/stable/
        getting_started/basics.html#dtypes)

    Returns: pandas dataframe
    '''
    return pd.read_csv(filepath, usecols=cols, dtype=col_types)

def show_distribution(df, variable):
    '''
    Graphs a histogram and the approximate distribtion of one variable in a
    dataframe.

    Inputs:
    df (pandas dataframe): dataframe containing the variable to show the
        distribution of as a column
    variable (str): the variable to show the distribution of; must be the name
        of a column in the dataframe

    Returns: matplotlib figure
    
    Replace density with box and whisker?

    Citations:
    Locating is_numeric_dtype: https://stackoverflow.com/questions/19900202/
    '''
    sns.set()
    if pd.api.types.is_numeric_dtype(df[variable]):
        f, (ax1, ax2) = plt.subplots(2, 1)
        sns.distplot(df[variable], kde=False, ax=ax1)
        sns.distplot(df[variable], hist=False, kde_kws={'shade': True},
                     rug=True, ax=ax2)
        ax1.set_title('Histogram')
        ax1.set_ylabel('Count')
        ax2.set_title('Estimated density') #change to box plot
    else:
        f, ax = plt.subplots(1, 1)
        val_counts = df[variable].value_counts()
        sns.barplot(x=val_counts.index, y=val_counts.values, ax=ax)
        ax.set_ylabel('Count')

    f.suptitle('Distribution of {}'.format(variable))
    f.subplots_adjust(hspace=.5, wspace=.5)

    return f

def pw_correlate(df, variables=None, visualize=False):
    '''
    Calculates a table of pairwise correlations between numberic variables.

    Inputs:
    df (pandas dataframe): dataframe containing the variables to calculate
        pairwise correlation between
    variables (list of strs): the list of variables to calculate pairwise
        correlations between; each passed str must be name of a numeric type
        (including booleans) column in the dataframe; default is all numberic
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
        sns.heatmap(corr_table, annot=True, fmt='.2f', linewidths=0.5, vmin=0,
                    vmax=1, square=True, cmap='coolwarm', ax=ax)

        labels = ['-\n'.join(wrap(l.get_text(), 15)) for l in ax.get_yticklabels()]
        ax.set_yticklabels(labels)
        labels = ['-\n'.join(wrap(l.get_text(), 15)) for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', rotation=0, labelsize='small')
        ax.tick_params(axis='x', rotation=90, labelsize='small')

        f.suptitle('Correlation Table')
        f.tight_layout()
        f.show()

    return corr_table

def summarize_data(df, grouping_vars=None, agg_cols=None, 
                   agg_funcs=[np.mean, np.var, np.median]):
    '''
    Groups rows based on the set of grouping variables and report summary
    statistics over the other numeric variables.

    Inputs:
    df (pandas dataframe): dataframe containing the variables to calculate
        pairwise correlation between
    grouping_vars (str or list of strs): optional variable or list of variables
        to group on before aggregating; each passed str must be name of a column
        in the dataframe; if not included, no grouping is performed
    agg_cols (str or list of strs): the variable or list of variables to
        aggregate after grouping; each passed str must be name of a column in
        the dataframe; default is all numeric type variables in the dataframe
    agg_funcs (list of functions): optional list of fuctions to aggregate
        with; default is mean, variance, and median

    Returns: pandas dataframe
    '''
    if not agg_cols:
        agg_cols = [col for col in df.columns
                        if (pd.api.types.is_numeric_dtype(df[col]) and
                            col not in grouping_vars)]
    if grouping_vars:
        summary = df.groupby(grouping_vars)\
                    [agg_cols]\
                    .agg(agg_funcs)
    else:
        summary = df[agg_cols]\
                    .agg(agg_funcs)

    return summary

def find_oulier_univariate(series, visualize=False):
    '''
    Identifies values in a series that fall more than 1.5 * IQR below the first
    quartile or 1.5 * IQR above the third quartile.

    Inputs:
    series (pandas series): the series to look for outliers in, must be numeric

    Returns: pandas series
    '''
    quartiles = np.percentile(series.dropna(), [0.25, 0.75])
    iqr = quartiles[1] - quartiles[0]
    lower_bound = quartiles[0] - iqr
    upper_bound = quartiles[1] + iqr

    return (lower_bound > series) | (upper_bound < series)

def find_outliers(df, excluded=None):
    '''
    Identifies outliers for each numeric column in a dataframe, and returns a
    data matching each record with the columns for which it is an outlier
    and the number and percent of numeric columns for which a is an outlier.
    Outlier is defined as any value thats fall more than 1.5 * IQR below the
    first quartile or 1.5 * IQR above the third quartile of all the values in a
    column.

    Inputs:
    df (pandas dataframe): the dataframe to find outliers in
    excluded (str list of strs): the name or a list of names of columns not to
        look for outliers in

    Returns: pandas series
    '''
    if not excluded:
        excluded = []

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    outliers = df[numeric_cols]\
                 .drop(excluded, axis=1, errors='ignore')\
                 .apply(find_oulier_univariate, axis=0)
    outliers['Count Outlier'] = outliers.sum(axis=1, numeric_only=True)
    outliers['% Outlier'] = (outliers['Count Outlier'] /
                             (len(outliers.columns) - 1) * 100)

    return outliers

def replace_missing(series):
    '''
    Replaces missing values in a series with the median value if the series is
    numeric and with the modal value if the series is non-numeric

    Inputs:
    series (pandas series): the series to replace data in

    Returns (pandas series):
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
    fields and the modal value in non-numeric columns

    Inputs:
    df (pandas dataframe): contains the data to preprocess

    Returns: pandas dataframe

    To-do: mention why preprocessing should be done prior to any variable
        transforms
    '''
    return df.apply(replace_missing, axis=0)

def cut_variable(series, bins, labels=None):
    '''
    Discretizes a continuous variable into bins. Bins are half-closed, [a, b).

    Inputs:
    series (pandas series): the variable to discretize
    bins (int or list of numerics): the binning rule:
        - if int: cuts the variable into n equally sized bins; the range of the
          variable is .1% on the either side to include the maximum and minimum
        - if list of numerics: cuts the variable into bins with edges determined
          by the sorted numerics in the list; any values not covered by the
          specified will be labeled as missing
    labels (list of str): optional list of labels for the bins; must be the same
        length as the number of bins specified

    Return: pandas series

    Possibly replace with pd.cut()?
    '''
    if type(bins) is int:
        min_val = min(series)
        max_val = max(series)
        range_size = max_val - min_val
        min_val = min_val - range_size * 0.001
        max_val = max_val + range_size * 0.001
        bins = np.linspace(min_val, max_val, num=bins + 1)

    cut = pd.Series(index=series.index)
    if labels:
        assert len(labels) == len(bins) - 1, ('You must specify the same ' +
                                              'number of labels and bins.')

    for i in range(len(bins) - 1):
        lb = bins[i]
        ub = bins[i + 1]
        if labels:
            cut[(lb <= series) & (series < ub)] = labels[i]
        else:
            cut[(lb <= series) & (series < ub)] = "[{0:.3f} to {1:.3f})".format(lb, ub)

    return cut

def create_dummies(df, columns=None):
    '''
    Transforms a variable into a set of dummy variables.

    Inputs:
    series (pandas dataframe/series): the data to transform dummies in
    columns (list of strs): optional list of column names containing categorical
        variables to convert to dummy variables; if not specified then any
        colums with dtype object or category are converted

    Returns: pandas dataframe where columns to be converted are replaced with
        columns contatining dummy variables

    Revert to bespoke to hadle NaNs? If so rewrite bespoke to meet this
    specification?
    '''
    return pd.get_dummies(df, columns=columns)

def generate_decision_tree(features, target, dt=None):
    '''
    Generates a decision tree to predict a target attribute (target) based on
    other attributes (features).

    Inputs:
    features (pandas dataframe): the features to build the decision tree with;
        all columns must be numeric (either the data within them should be
        numeric or it should be converted from categorical data to dummies)
    target (pandas series): the target the decision tree is designed to predict;
        should be categorical data
    dt (sklearn.tree.DecisionTree): optional DecisionTreeClassifier object so
        the parameters of the DecisionTreeClassifier can be specified; if
        unspecified, a new DecisionTreeClassifier object will be instantiated
        with all the default arguments, except the function to measure the
        quality of a split ('entropy' will be specified instead of the default,
        'gini')

    Returns: sklearn.tree.DecisionTreeClassifier, the trained 
        DecisionTreeClassifier 

    Citations:
    DecisionTreeClassifier docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.DecisionTreeClassifier.html#
        sklearn.tree.DecisionTreeClassifier
    '''
    if not dt:
        dt = tree.DecisionTreeClassifier(criterion='entropy')

    return dt.fit(features, target)

def score_decision_tree(test_features, test_target, dt):
    '''
    Returns the mean accuracy of the decision tree's predictions on a set of
    test data.

    Inputs:
    test_features (pandas dataframe): the feature values for the observations
        the decision tree is being tested against; all columns must be numeric
        (either the data within them should be numeric or it should be converted
        from categorical data to dummies)
    target (pandas series): the target attribute values for the observations the
        the decision tree is being tested against; should be categorical data
    dt (sklearn.tree.DecisionTreeClassifier): a trained decision tree classifier
        model

    Returns: float

    Citations:
    DecisionTreeClassifier docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.DecisionTreeClassifier.html#
        sklearn.tree.DecisionTreeClassifier
    '''
    return dt.score(test_features, test_target)

def visualize_decision_tree(dt, feature_names, class_names, filepath='pdf'):
    '''
    Saves and pens a PDF visualizing the specified decision tree.

    Inputs:
    dt (sklearn.tree.DecisionTreeClassifier): a trained decision tree classifier
    feature_names (list of strs): a list of the features the data was trained
        with; must be in the same order as the features in the dataset
    class_names (list of strs): a list of the classes of the target attribute
        the model is predicting; must be in the same order as the features in
        the dataset
    filepath (str): optitional parameter specifying the output path for the
        visualization (do not include the file extension); default is 'tree' in
        the present working directory
    
    Citations:
    Guide to sklearn decision trees: https://scikit-learn.org/stable/modules/
        tree.html
    sklearn.tree.export_graphviz docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.export_graphviz.htm
    '''
    class_names.sort()
    dot_data = tree.export_graphviz(dt, None, feature_names=feature_names, 
                                  class_names=class_names, filled=True)
    graph = graphviz.Source(dot_data)
    output_path = graph.render(filename=filepath, view=True)
