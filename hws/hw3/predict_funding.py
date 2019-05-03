'''
Predicting Donors Choose Funding

Ben Fogarty

2 May 2019
'''

import sys
from sklearn import tree
import numpy as np
import pandas as pd
import pipeline_library as pl
import matplotlib.pyplot as plt

def go(file):
    '''
    Applies the pipeline library to predicting if a project on Donors Choose
    will not get full funding within 60 days.

    Inputs:
    filepath (str): path to the file containing the training data
    '''
    col_types = {'projectid': str,
                 'teacher_acctid': str,
                 'schoolid': str,
                 'school_ncesid': str,
                 'school_latitude': float,
                 'school_longitude': float,
                 'school_city': str,
                 'school_state': str,
                 'school_metro': str,
                 'school_district': str,
                 'school_county': str,
                 'school_charter': str,
                 'school_magnet': str,
                 'teacher_prefix': str,
                 'primary_focus_subject': str,
                 'primary_focus_area': str,
                 'secondary_focus_subject': str,
                 'secondary_focus_area': str,
                 'resource_type': str,
                 'poverty_level': str,
                 'grade_level': str,
                 'total_price_including_optional_support': float,
                 'students_reached': float,
                 'eligible_double_your_impact_match': str,
                 'date_posted': str,
                 'datefullyfunded': str}
    df = pl.read_csv(file, col_types=col_types, index_col='projectid')
    df = transform_data(df)

    explore_data(df)
    df = preprocess_data(df)
    df = generate_features(df)
    training, testing = pl.create_temporal_splits(df, 'date_posted', {'months': 6})
    del(df) #full df no longer needed after splits

    for i in range(len(training)):
        training[i] = training[i].drop('date_posted', axis=1)
        testing[i] = testing[i].drop('date_posted', axis=1)
    
    models = [{'model': 'dt',
               'criterion': 'entropy',
               'max_depth': 35},
               {'model': 'lr',
                'solver': 'sag'},
               {'model': 'svc'},
               {'model': 'rf',
                'criterion': 'entropy',
                'max_depth': 35,
                'n_estimators': 10},
               {'model': 'boosting',
                'n_estimators': 10},
               {'model': 'bagging',
                'base_estimator': tree.DecisionTreeClassifier(max_depth=35)},
               {'model': 'knn',
                'n_neighbors': 10},
               {'model': 'dummy',
                'strategy': 'most_frequent'}]
    classifiers = train_classifiers(models, training)
    evaluate_classifiers(classifiers, models, testing)


def transform_data(df):
    '''
    Changes the types of columns in the dataset and creates new columns to
    allow for better data exploration and modeling.

    Inputs:
    df (pandas dataframe): the dataset

    Returns: pandas dataframe
    '''
    df['date_posted'] = pd.to_datetime(df.date_posted)
    df['datefullyfunded'] = pd.to_datetime(df.datefullyfunded)
    df['daystofullfunding'] = pl.create_time_diff(df.date_posted,
                                                  df.datefullyfunded)
    df['daystofullfunding'] = df.daystofullfunding.apply(lambda x: x.days)

    tf_cols = ['school_charter', 'school_magnet', 'eligible_double_your_impact_match']    
    for col in tf_cols:
        df[col] = (df[col] == 't').astype('float')

    return df

def explore_data(df):
    '''
    Generates distributions of variables, correlations, outliers, and other
    summaries of variables in the dataset.

    Inputs:
    df (pandas dataframe): the dataframe
    '''
    print('---------------------\n| General statistics |\n---------------------')
    n_projects = len(df)
    print('The dataset contains {} projects.\n'.format(n_projects))
    
    n_teachers = len(df.teacher_acctid.value_counts())
    n_schools = len(df.schoolid.value_counts())
    print('These projects were submitted from {} different'.format(n_teachers) + 
          '\nteacher accounts across {} different schools.\n'.format(n_schools))

    n_positive = np.sum(df.daystofullfunding <= 60)
    p_positive = n_positive / n_projects * 100
    print('{0} projects ({1:.1f}%) '.format(n_positive, p_positive) +
          'were funded within sixty days of posting.\n')
    print()


    per_teach_desc, per_teach_fig = pl.count_per_categorical(df, 'teacher_acctid')
    print(per_teach_desc)
    per_teach_fig.show()
    print()

    per_dist_desc, per_dist_fig = pl.count_per_categorical(df, 'school_district')
    print(per_dist_desc)
    per_dist_fig.show()
    print()
    
    per_school_desc, per_school_fig = pl.count_per_categorical(df, 'schoolid')
    print(per_school_desc)
    per_school_fig.show()
    print()

    print(pl.summarize_data(df, agg_cols=['daystofullfunding']))
    pl.show_distribution(df.daystofullfunding).show()
    print()

    print('---------------------\n|  Location splits   |\n---------------------')
    pl.show_distribution(df.school_metro).show()

    location_cols = ['school_metro', 'school_state']
    for col in location_cols:
        print(pl.summarize_data(df, grouping_vars=[col],
                                agg_cols=['daystofullfunding']))
    print()

    print('---------------------\n| School type splits |\n---------------------')
    type_cols = ['school_charter', 'school_magnet', 'poverty_level']
    for col in type_cols:
        pl.show_distribution(df[col].astype(str)).show()
        print(pl.summarize_data(df, grouping_vars=[col],
                                agg_cols=['daystofullfunding']))
    print()

    print('----------------------\n| Project type splits  |\n----------------------')
    pl.show_distribution(df.primary_focus_area).show()
    print(pl.summarize_data(df, grouping_vars=['primary_focus_area',
                                               'secondary_focus_area'],
                            agg_cols=['daystofullfunding']))
    print()

    pl.show_distribution(df.primary_focus_subject).show()
    print(pl.summarize_data(df, grouping_vars=['primary_focus_subject', 
                                               'secondary_focus_subject'],
                            agg_cols=['daystofullfunding']))
    print()

    pl.show_distribution(df.grade_level).show()
    print(pl.summarize_data(df, grouping_vars=['grade_level'], 
                            agg_cols=['daystofullfunding']))
    print()

    pl.show_distribution(df.resource_type).show()
    print(pl.summarize_data(df, grouping_vars=['resource_type'], 
                            agg_cols=['daystofullfunding']))
    print()

    pl.show_distribution(df.eligible_double_your_impact_match.astype(str))\
      .show()
    print(pl.summarize_data(df, grouping_vars=['eligible_double_your_impact_match'], 
                            agg_cols=['daystofullfunding']))

    print('----------------------\n| Numeric variables |\n----------------------')
    pl.show_distribution(df.total_price_including_optional_support).show()
    pl.show_distribution(df.students_reached).show()
    print()

    print(pl.summarize_data(df, agg_cols=['total_price_including_optional_support',
                                          'students_reached']))
    print()

    print('----------------------\n| Numeric correlations |\n----------------------')
    print(pl.pw_correlate(df.drop(['school_latitude', 'school_longitude'], axis=1), 
                          visualize=True))

    print('----------------------\n| Outliers & missing data |\n----------------------')
    print(pl.report_n_missing(df))
    plt.show()

def preprocess_data(df):
    '''
    Preprocesses the data

    Inputs:
    df (pandas dataframe): the dataset

    Returns: pandas dataframe
    '''
    df['secondary_focus_area'] = df.secondary_focus_area.fillna('N/A')
    df['secondary_focus_subject'] = df.secondary_focus_subject.fillna('N/A')
    df = pl.preprocess_data(df)

    return df

def generate_features(df):
    '''
    Generates categorical and binary features.

    Inputs:
    df (pandas dataframe): the dataframe

    Returns: pandas dataframe, the dataset after generating features
    '''
    df = df.drop(['school_longitude', 'school_latitude', 'schoolid',
                  'teacher_acctid', 'school_district', 'school_ncesid',
                  'school_county'],
                  axis=1)
    
    numeric_cols = ['students_reached', 'total_price_including_optional_support']
    for col in numeric_cols:
        df[col] = pl.cut_variable(df[col], 10, 
                                  labels=['dec' + str(i / 10) for i in range(10)])

    cat_cols = ['school_city', 'school_state', 'school_metro',
                'teacher_prefix',
                'primary_focus_subject', 'primary_focus_area',
                'secondary_focus_subject', 'secondary_focus_area',
                'resource_type',
                'poverty_level', 'grade_level'] + numeric_cols
    df = pl.create_dummies(df, cat_cols)

    df['fully_funded_60days'] = df.daystofullfunding > 60
    df = df.drop(['daystofullfunding', 'datefullyfunded'], axis=1)
    return df

def train_classifiers(models, training):
    '''
    Returns a 2-D list that where where each inner list is a set of
    classifiers and the outer list represents each training/test set (i.e.
    at location 0,0 in the output list is the first model trained on the 
    first set and at location 1,0 is the first model trained on the second
    set).

    Inputs:
    models (dict): specifications for the classifiers
    training (list of pandas dataframe): a list of training datasets

    Returns: 2D list of trained sklearn classifiers
    '''
    classifiers = []
    for i in range(len(training)):
        print('----------\nBuilding with training set {}\n----------'.format(i + 1))
        features = training[i].drop('fully_funded_60days', axis=1)
        target = training[i].fully_funded_60days
        classifiers.append(pl.generate_classifiers(features, target, models))

    return classifiers

def evaluate_classifiers(classifiers, models, testing):
    '''
    Prints out evaluations for the trained models using the specified testing
    datasets

    Inputs:
    classifiers (2d list of trained sklearn classifier objects): trained
        classifiers as output by train_models
    models (dict): model specifications used to generate the classifiers
    testing (list of pandas dataframes): the testing datasets associated with
        each outer list in models
    '''
    for i in range(len(models)):
        print('---------\n {} \n---------'.format(models[i]['model']))
        table = pd.DataFrame()
        for j in range(len(testing)):
            features = testing[j].drop('fully_funded_60days', axis=1)
            y_actual = testing[j].fully_funded_60days
            table['Test/Training Set {}'.format(j + 1)], fig =\
                pl.evaluate_classifier(classifiers[j][i], features, y_actual,\
                [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50], models[i]['model'],\
                'Test/Training Set {}'.format(j + 1))
            fig.show()
        print(table)
        plt.show()

if __name__ == '__main__':
    usage = "python3 predict_funding.py <dataset> <parameters>"
    filepath = sys.argv[1]
    go(filepath)
