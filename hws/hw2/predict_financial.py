'''
Simple Machine Learning Pipeline

Ben Fogarty

18 April 2018
'''

import pandas as pd
import pipeline_library as pl

def go(filepath):
    '''
    <update>

    Inputs:
    filepath (str): path to the file containing the training data
    '''
    col_types = {'SeriousDlqin2yrs': float,
                 'RevolvingUtilizationOfUnsecuredLines': float,
                 'age': float,
                 'NumberOfTime30-59DaysPastDueNotWorse': float,
                 'zipcode': str,
                 'DebtRatio': float,
                 'MonthlyIncome': float,
                 'NumberOfOpenCreditLinesAndLoans': float,
                 'NumberOfTimes90DaysLate': float,
                 'NumberRealEstateLoansOrLines': float,
                 'NumberOfTime60-89DaysPastDueNotWorse': float,
                 'NumberOfDependents': float}
    df = pl.read_csv(filepath, col_types=col_types)
    df = pl.preprocess_data(df)
    explore_data(df)
    df = process_data(df)
    dt, accuracy = build_eval_model(df)
    print('Accuracy: {}'.format(accuracy))

    print(df.columns)
    feature_names = list(df.columns)
    print(feature_names)
    feature_names.remove('PersonID')
    feature_names.remove('SeriousDlqin2yrs')
    pl.visualize_decision_tree(dt, feature_names, class_names=['0', '1'])

def explore_data(df):
    '''
    Generates distributions of variables, correlations, outliers, and other 
    summaries of variables in the dataset.

    Inputs:
    df (pandas dataframe): the dataframe
    '''
    summary = pl.summarize_data(df)

    pw_corr_vars = ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines',
       'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
       'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    correlations = pl.pw_correlate(df, variables=pw_corr_vars)

    for var in df.columns:
        if not var == 'PersonID':
            print(var)
            if var in summary.index:
                print(summary.loc[var, :])
            pl.show_distribution(df, var).show()
            if var in pw_corr_vars:
                print('Correlations with {}'.format(var))
                print(correlations[var])
            print()

    print('By Zipcode Summary:')
    print(pl.summarize_data(df, grouping_vars='zipcode'))

    print('Outlier summary:')
    outliers = pl.find_outliers(df, excluded=['SeriousDlqin2yrs', 'PersonID'])
    outliers.sort_values('Count Outlier', ascending=False, inplace=True)
    print(outliers['Count Outlier'].value_counts())

def process_data(df):
    '''
    Generates categorical and binary features.

    Inputs:
    df (pandas dataframe): the dataframe

    Returns: pandas dataframe, the processed dataset
    '''
    df = pl.create_dummies(df, 'zipcode')

    return df


def build_eval_model(df):
    '''
    Builds a decision tree model to predict finanical distress within the next
    two years and assesses the model's accuracy the mean accuracy of the
    decision tree's predictions on a set of test data.

    Inputs:
    df: the dataframe

    Returns: tuple of sklearn.tree.DecisionTreeClassifier (the trained decision
        tree) and float 

    For later implementation: splitting data into test and training sets
    '''
    features = df.drop(['PersonID', 'SeriousDlqin2yrs'], axis=1)
    target = df.SeriousDlqin2yrs
    dt = pl.generate_decision_tree(features, target)

    accuracy = pl.score_decision_tree(dt, features, target)

    return dt, accuracy


if __name__ == '__main__':
    go()