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
    grouping_vars (str or list of strs): optional variable or list of variables
        to group on before aggregating; each passed str must be name of a column
        in the dataframe; if not included, no grouping is performed
    agg_cols (str or list of strs): the variable or list of variables to
        aggregate after grouping; each passed str must be name of a column in
        the dataframe; default is all numeric type variables in the dataframe

    Returns: pandas dataframe
    '''
    if agg_cols:
        df = df[agg_cols]
    
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

def cut_variable(series, bins, labels=None):
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

    Return: pandas series
    '''
    if type(bins) is int:
        return pd.qcut(series, bins, labels=labels, duplicates='drop')\
                 .astype('category')

    return pd.cut(series, bins, labels=labels, include_lowest=True)\
             .astype('category')

def create_dummies(df, column):
    '''
    Transforms a variable into a set of dummy variables.

    Inputs:
    series (pandas dataframe/series): the data to transform dummies in
    column (list of strs): column name containing categorical
        variables to convert to dummy variables; if not specified then any
        colums with dtype object or category are converted

    Returns: pandas dataframe where the columns to be converted is replaced with
        columns containing dummy variables
    '''
    col = df[column]
    values = list(col.value_counts().index)
    output = df.drop(column, axis=1)
    for value in values:
        dummy_name = '{}_{}'.format(column, value)
        output[dummy_name] = (col == value)
        output.loc[col.isnull(), dummy_name] = float('nan')

    return output

def generate_decision_tree(features, target, dt=None):
    '''
    Generates a decision tree to predict a target attribute (target) based on
    other attributes (features).

    Inputs:
    features (pandas dataframe): data for features to build the decision tree 
        with; all columns must be numeric in type
    target (pandas series): data for target attribute the decision tree is
        designed to predict; should be categorical data with a numerial form
    dt (sklearn.tree.DecisionTree): optional DecisionTreeClassifier object so
        the parameters of the DecisionTreeClassifier can be specified; if
        unspecified, a new DecisionTreeClassifier object will be instantiated
        with all the default arguments

    Returns: sklearn.tree.DecisionTreeClassifier, the trained 
        DecisionTreeClassifier 

    Citations:
    DecisionTreeClassifier docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.DecisionTreeClassifier.html#
        sklearn.tree.DecisionTreeClassifier
    '''
    if not dt:
        dt = tree.DecisionTreeClassifier()

    return dt.fit(features, target)

def score_decision_tree(dt, test_features, test_target):
    '''
    Returns the mean accuracy of the decision tree's predictions on a set of
    test data.

    Inputs:
    dt (sklearn.tree.DecisionTreeClassifier): a trained decision tree classifier
        model
    test_features (pandas dataframe): testing data for the features in the
        decision tree; structure of the data (columns and column types) must
        match the data used to train the decision tree
    target (pandas series): testing data for the target attribute a decision
        tree is predicting; structure of the data (columns and column types) 
        must match the data used to train the decision tree

    Returns: float

    Citations:
    DecisionTreeClassifier docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.DecisionTreeClassifier.html#
        sklearn.tree.DecisionTreeClassifier
    '''
    return dt.score(test_features, test_target)

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
