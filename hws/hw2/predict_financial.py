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
    df = pl.read_csv(filepath)
    explore_data(df)
    process_data(df)
    dt, accuracy = build_model(df)

def explore_data(df):
    '''
    Generates distributions of variables, correlations, outliers, and other 
    summaries of variables in the dataset.

    Inputs:
    df (pandas dataframe): the dataframe
    '''
    pass

def process_data(df):
    '''
    Pre-processes data and generates categorical and binary features.

    Inputs:
    df (pandas dataframe): the dataframe

    Returns: pandas dataframe, the processed dataset
    '''
    pass

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
    pass

if __name__ == '__main__':
    go()