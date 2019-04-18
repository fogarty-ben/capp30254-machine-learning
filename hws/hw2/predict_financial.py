'''
Simple Machine Learning Pipeline

Ben Fogarty

18 April 2018
'''

import pandas as pd
import pipeline_library as pl
from sklearn import tree

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
    df = pl.read_csv('credit-data.csv', col_types=col_types, index_col='PersonID')
    explore_data(df)
    
    df = pl.preprocess_data(df)
    df = generate_features(df)
    
    target_col = 'SeriousDlqin2yrs'
    dt = build_eval_model(df, target_col)
    features_cols = df.drop(target_col, axis=1).columns
    pl.visualize_decision_tree(dt, features_cols, 
                               ['No Financial Distress', 'Financial Distress'])

    features = df.drop(target_col, axis=1)
    target = df[target_col]
    accuracy = pl.score_decision_tree(dt, features, target)
    print('Accuracy: {}'.format(accuracy))

def explore_data(df):
    '''
    Generates distributions of variables, correlations, outliers, and other 
    summaries of variables in the dataset.

    Inputs:
    df (pandas dataframe): the dataframe
    '''
    summary = pl.summarize_data(df)
    print(summary)
    
    correlate = pl.pw_correlate(df, visualize=True)
    print(correlate)

    by_zip = pl.summarize_data(df, grouping_vars='zipcode')
    print(by_zip.xs('mean', axis=0, level=1))

    for var in df.columns:
        pl.show_distribution(df[var].dropna()).show()

    outliers = pl.find_outliers(df)
    print(outliers['Count Outlier'].value_counts().sort_index())
    print(outliers.drop(['Count Outlier', '% Outlier'], axis=1).sum().sort_values())

def generate_features(df):
    '''
    Generates categorical and binary features.

    Inputs:
    df (pandas dataframe): the dataframe

    Returns: pandas dataframe, the processed dataset
    '''
    df = pl.create_dummies(df, 'zipcode')

    df.MonthlyIncome = pl.cut_variable(df.MonthlyIncome, 20)
    df = pl.create_dummies(df, 'MonthlyIncome')

    df.RevolvingUtilizationOfUnsecuredLines = pl.cut_variable(df.RevolvingUtilizationOfUnsecuredLines, 10)
    df = pl.create_dummies(df, 'RevolvingUtilizationOfUnsecuredLines')

    df['NumberOfTime30-59DaysPastDueNotWorse'] = pl.cut_variable(df['NumberOfTime30-59DaysPastDueNotWorse'], [0, 1, float('inf')], labels=['Zero', 'One or more'])
    df = pl.create_dummies(df, 'NumberOfTime30-59DaysPastDueNotWorse')

    df['NumberOfTime60-89DaysPastDueNotWorse'] = pl.cut_variable(df['NumberOfTime60-89DaysPastDueNotWorse'], [0, 1, float('inf')], labels=['Zero', 'One or more'])
    df = pl.create_dummies(df, 'NumberOfTime60-89DaysPastDueNotWorse')

    df['NumberOfTimes90DaysLate'] = pl.cut_variable(df['NumberOfTimes90DaysLate'], [0, 1, float('inf')], labels=['Zero', 'One or more'])
    df = pl.create_dummies(df, 'NumberOfTimes90DaysLate')

    return df


def build_eval_model(df, target_col):
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
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=15)
    
    features = df.drop(target_col, axis=1)
    target = df[target_col]

    dt = pl.generate_decision_tree(features, target, dt=dt)

    return dt


if __name__ == '__main__':
    go()