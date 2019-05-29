'''
Predicting Donors Choose Funding

Ben Fogarty

2 May 2019
'''

import argparse
import json
import sys
from sklearn import tree
import numpy as np
import pandas as pd
import pipeline_library as pl
import matplotlib.pyplot as plt

def apply_pipeline(dataset, preprocessing, features, models, seed=None):
    '''
    Applies the pipeline library to predicting if a project on Donors Choose
    will not get full funding within 60 days.

    Inputs:
    filepath (str): path to the file containing the training data
    seed (str): seed used for random process to adjucate ties when translating
        predicted probabilities to predicted classes given some percentile
        threshold
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

    #explore_data(df)
    training_splits, testing_splits = pl.create_temporal_splits(df, 
                                      'date_posted', {'months': 6}, gap={'days': 60})
    for i in range(len(training_splits)):
        training_splits[i] = preprocess_data(training_splits[i])
        testing_splits[i] = preprocess_data(testing_splits[i])

        training_splits[i], testing_splits[i] = generate_features(training_splits[i],
                                                                  testing_splits[i])
    for model in models:
        print('-' * 20 +  '\nModel Specifications\n' + str(model) + '\n' + '_' * 20)
        model_name = model.get('name', None)
        trained_classifiers = train_classifiers(model, training_splits)
        pred_probs = predict_probs(trained_classifiers, testing_splits)
        evaluate_classifiers(pred_probs, testing_splits, seed, model_name)


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
    methods = {'secondary_focus_area': 'manual',
               'secondary_focus_subject': 'manual'}
    manual_vals = {'secondary_focus_area': 'N/A',
                   'secondary_focus_subject': 'N/A'}
    df = pl.preprocess_data(df, methods=methods, manual_vals=manual_vals)

    return df

def generate_features(training, testing, n_ocurr_cols, scale_cols, bin_cols,
                      cat_cols,binary_cut_cols, drop_cols):
    '''
    Generates categorical, binary, and scaled features. While features are
    generate for the training data independent of the testing data, features
    for the testing data sometimes require ranges or other information about the
    properties of features created for the training data to ensure consistency.

    Inputs:
    training (pandas dataframe): the training data
    testing (pandas dataframe): the testing data

    Returns: tuple of pandas dataframe, the training and testing datasets after
        generating the features
    '''
    '''
    df = df.drop(['school_longitude', 'school_latitude', 'schoolid',
                  'teacher_acctid', 'school_district', 'school_ncesid',
                  'school_county'],
                  axis=1)
    '''
    for col in n_ocurr_cols:
        training.loc[:, str(col) + '_n_occur'] = pl.generate_n_occurences(training[col])
        testing.loc[:, str(col) + '_n_occur'] = pl.generate_n_occurences(testing[col],
                                                             addl_obs=training[col])

    scale_cols = scale_cols + [col + '_n_occur' for col in n_ocurr_cols]
    for col in scale_cols:
        max_training = max(training[col])
        min_training = min(training[col])
        training.loc[:, col] = pl.scale_variable_minmax(training[col])
        testing.loc[:, col] = pl.scale_variable_minmax(testing[col], a=max_training,
                                                b=min_training)

    bin_cols = []
    for col in bin_cols:
        training.loc[:, col], bin_edges = pl.cut_variable(training[col], n_quantlies)
        bin_edges[0] = - float('inf') #test observations below the lowest observation
        #in the training set should be mapped to the lowest bin
        bin_edges[-1] = float('inf') #test observations above the highest observation
        #in the training set should be mapped to the highest bin
        testing[col], _ = pl.cut_variable(testing[col], bin_edges)


    cat_cols = ['school_city', 'school_state', 'school_metro',
                'teacher_prefix',
                'primary_focus_subject', 'primary_focus_area',
                'secondary_focus_subject', 'secondary_focus_area',
                'resource_type',
                'poverty_level', 'grade_level'] + bin_cols
    for col in cat_cols:
        values = list(training[col].value_counts().index)
        training = pl.create_dummies(training, col, values=values)
        testing = pl.create_dummies(testing, col, values=values)

    training['fully_funded_60days'] = pl.cut_binary(training.daystofullfunding, 60)
    testing['fully_funded_60days'] = pl.cut_binary(testing.daystofullfunding, 60)

    drop_cols = ['school_longitude', 'school_latitude', 'schoolid',
                  'teacher_acctid', 'school_district', 'school_ncesid',
                  'school_county', 'daystofullfunding', 'datefullyfunded',
                  'date_posted']
    drop_cols = drop_cols + [str(col) + '_missing' for col in drop_cols]
    training = training.drop(drop_cols, axis=1)
    testing = testing.drop(drop_cols, axis=1)
    
    return training, testing

def train_classifiers(model, training):
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
        print('Building with training set {}'.format(i + 1))
        features = training[i].drop('fully_funded_60days', axis=1)
        target = training[i].fully_funded_60days
        classifiers.append(pl.generate_classifier(features, target, model))

    return classifiers

def predict_probs(trained_classifiers, testing_splits):
    '''
    Generates predictions for the observations in the i-th training split based
    on the i-th trained classifier.

    Inputs:
    trained_classifiers (list of sklearn classifers): the i-th model should have
        been trained on the i-th sklearn training split
    testing_splits (list of pandas dataframe): the i-th testing split should be
        associated with the i-th training split

    Returns: list of pandas series
    '''
    pred_probs = []
    for i in range(len(trained_classifiers)):
        print('Predicting probabilies with training set {}'.format(i+1))
        features = testing_splits[i].drop('fully_funded_60days', axis=1)
        pred_probs.append(pl.predict_target_probability(trained_classifiers[i],
                                                        features))

    return pred_probs

def evaluate_classifiers(pred_probs, testing_splits, seed=None, model_name=None):
    '''
    Prints out evaluations for the trained model using the specified testing
    datasets

    Inputs:
    pred_probs (list of pandas series): list of predicted probabilities
        generated by some classifier; the i-th series of predicted probabilities
        should be associated with the i-th training split
    testing_splits (list of pandas dataframe): the i-th testing split should be
        associated with the i-th series of predicted probabilities
    seed (str): seed used for random process to adjucate ties when translating
        predicted probabilities to predicted classes given some percentile
        threshold
    model_name (str): model name to include in the title of the 
        precision/recall curve graph
    '''
    table = pd.DataFrame()
    for i in range(len(pred_probs)):
        print('Evaluating predictions with training set {}'.format(i+1))
        y_actual = testing_splits[i].fully_funded_60days
        table['Test/Training Set {}'.format(i + 1)], fig =\
            pl.evaluate_classifier(pred_probs[i], y_actual,\
            [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50], seed=seed, 
            model_name=model_name,
            dataset_name='Training/Testing Set # {}'.format(i + 1))
        fig.show()
    print(table)
    plt.show()

def parse_args(args):
    '''
    Parses command line arguments for use by the rest of the software

    Inputs:
    args (argsparse Namespace): arguments from the command line

    Returns: 5-ple of filepath to dataset (str), pre-procesing specs (dict),
    feature generation specs (dict), model specs (list of dicts), seed (int)
    '''
    print(args)
    dataset_fp = args.dataset

    if args.preprocess is not None:
        with open(args.preprocess, 'r') as file:
            preprocess_specs = json.load(file)
    else:
        preprocess_specs = {}

    with open(args.features, 'r') as file:
        feature_specs = json.load(file)

    with open(args.models, 'r') as file:
        model_specs = json.load(file)

    seed = args.seed

    return dataset_fp, preprocess_specs, feature_specs, model_specs, seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Apply machine learning" +
        "pipeline to Donors' Choose Data"))
    parser.add_argument('-d', '--data', type=str, dest='dataset', required=True,
                        help="Path to the Donors' Choose dataset")
    parser.add_argument('-f', '--features', type=str, dest='features',
                        required=True, help="Path to the features config JSON")
    parser.add_argument('-m', '--models', type=str, dest='models',
                        required=True, help="Path to the model specs JSON")
    parser.add_argument('-p', '--preprocess', type=str, dest='preprocess',
                        required=False, help="Path to the preprocessing config JSON")
    parser.add_argument('-s', '--seed', type=float, dest='seed', required=False,
                        help='Random seed for tiebreaking when predicting classes')
    args = parser.parse_args()
    
    data, preprocess, features, models, seed = parse_args(args)
    apply_pipeline(data, preprocess, features, models, seed)
    '''
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

    feature_args = {'n_ocurr_cols': ['schoolid', 'teacher_acctid',
                                     'school_district', 'school_county'],
                    'scale_cols': ['students_reached', 
                                   'total_price_including_optional_support',
                                   'schoolid_n_occur', 'teacher_acctid_n_occur',
                                   'school_district_n_occur',
                                   'school_county_n_occur'],
                    'bin_cols': {},
                    'cat_cols': ['school_city', 'school_state', 'school_metro',
                                 'teacher_prefix', 'primary_focus_subject',
                                 'primary_focus_area', 'secondary_focus_subject',
                                 'secondary_focus_area', 'resource_type',
                                 'poverty_level', 'grade_level'],
                    'binary_cut_cols': {'daystofullfunding', {'threshold': 60}},
                    'drop_cols': ['school_longitude', 'school_latitude',
                                  'schoolid', 'teacher_acctid',
                                  'school_district', 'school_ncesid',
                                  'school_county', 'daystofullfunding',
                                  'datefullyfunded','date_posted',
                                  'school_longitude_missing',
                                  'school_latitude_missing',
                                  'schoolncesid_missing',
                                  'daystofullfunding_missing',
                                  'datefullyfunded_missing',
                                  'date_posted_missing']}

    preprocessing_args = {}
    '''
