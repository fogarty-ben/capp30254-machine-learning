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
    dataset (str): path to the file containing the training data
    preprocessing (dict): dictionary of keyword arguments to pass to the
        preprocess_data function
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
    df = pl.read_csv(dataset, col_types=col_types, index_col='projectid')
    df = transform_data(df)

    explore_data(df)
    training_splits, testing_splits = pl.create_temporal_splits(df, 
                                      'date_posted', {'months': 6}, gap={'days': 60})
    for i in range(len(training_splits)):
        training_splits[i] = preprocess_data(training_splits[i], **preprocessing)
        testing_splits[i] = preprocess_data(testing_splits[i], **preprocessing)

        training_splits[i], testing_splits[i] = generate_features(training_splits[i],
                                                                  testing_splits[i],
                                                                  **features)
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
    df['not_fully_funded_60days'] = df.daystofullfunding > 60

    tf_cols = ['school_charter', 'school_magnet', 'eligible_double_your_impact_match']    
    for col in tf_cols:
        df[col] = (df[col] == 't').astype('float')

    return df

def explore_data(df, save_figs):
    '''
    Generates distributions of variables, correlations, outliers, and other
    summaries of variables in the dataset.

    Inputs:
    df (pandas dataframe): the dataframe
    save_figs (bool): if true, figures are saved and not displayed; if false,
        figures are dispalyed
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
    if save_fig:
        plt.savefig('teacher_acctid.png')
    else:
        plt.show()
    print()

    per_dist_desc, per_dist_fig = pl.count_per_categorical(df, 'school_district')
    print(per_dist_desc)
    if save_fig:
        plt.savefig('school_district.png')
    else:
        plt.show()
    print()
    
    per_school_desc, per_school_fig = pl.count_per_categorical(df, 'schoolid')
    print(per_school_desc)
    if save_fig:
        plt.savefig('schoolid.png')
    else:
        plt.show()
    print()

    print(pl.summarize_data(df, agg_cols=['daystofullfunding']))
    fullfunding_fig = pl.show_distribution(df.daystofullfunding)
    if save_fig:
        plt.savefig('daystofullfunding.png')
    else:
        plt.show()
    print()

    print('---------------------\n|  Location splits   |\n---------------------')
    school_metro_fig = pl.show_distribution(df.school_metro)
    if save_fig:
        plt.savefig('school_metro.png')
    else:
        plt.show()

    location_cols = ['school_metro', 'school_state']
    for col in location_cols:
        print(pl.summarize_data(df, grouping_vars=[col],
                                agg_cols=['daystofullfunding']))
    print()

    print('---------------------\n| School type splits |\n---------------------')
    type_cols = ['school_charter', 'school_magnet', 'poverty_level']
    for col in type_cols:
        school_type_fig = pl.show_distribution(df[col].astype(str))
        if save_fig:
            plt.savefig(col + '.png')
        else:
            plt.show()
        print(pl.summarize_data(df, grouping_vars=[col],
                                agg_cols=['daystofullfunding']))
    print()

    print('----------------------\n| Project type splits  |\n----------------------')
    primary_focus_area_fig = pl.show_distribution(df.primary_focus_area)
    if save_fig:
        plt.savefig('primaryfocusarea.png')
    else:
        plt.show()
    print(pl.summarize_data(df, grouping_vars=['primary_focus_area',
                                               'secondary_focus_area'],
                            agg_cols=['daystofullfunding']))
    print()

    primary_focus_subj_fig = pl.show_distribution(df.primary_focus_subject)
    if save_fig:
        plt.savefig('primaryfocussubject.png')
    else:
        plt.show()
    print(pl.summarize_data(df, grouping_vars=['primary_focus_subject', 
                                               'secondary_focus_subject'],
                            agg_cols=['daystofullfunding']))
    print()

    grade_level_fig = pl.show_distribution(df.grade_level)
    if save_fig:
        plt.savefig('gradelevel.png')
    else:
        plt.show()
    print(pl.summarize_data(df, grouping_vars=['grade_level'], 
                            agg_cols=['daystofullfunding']))
    print()

    pl.show_distribution(df.resource_type).show()
    print(pl.summarize_data(df, grouping_vars=['resource_type'], 
                            agg_cols=['daystofullfunding']))
    print()

    double_fig = pl.show_distribution(df.eligible_double_your_impact_match.astype(str))
    if save_fig:
        plt.savefig('doubleimpact.png')
    else:
        plt.show()
    print(pl.summarize_data(df, grouping_vars=['eligible_double_your_impact_match'], 
                            agg_cols=['daystofullfunding']))

    print('----------------------\n| Numeric variables |\n----------------------')
    price_fig = pl.show_distribution(df.total_price_including_optional_support)
    if save_fig:
        plt.savefig('totalprice.png')
    else:
        plt.show()
    students_reached_fig = pl.show_distribution(df.students_reached)
    if save_fig:
        plt.savefig('studentsreached.png')
    else:
        plt.show()
    print()

    print(pl.summarize_data(df, agg_cols=['total_price_including_optional_support',
                                          'students_reached']))
    print()

    print('----------------------\n| Numeric correlations |\n----------------------')
    print(pl.pw_correlate(df.drop(['school_latitude', 'school_longitude'], axis=1), 
                          visualize=True))

    print('----------------------\n| Outliers & missing data |\n----------------------')
    print(pl.report_n_missing(df))

def preprocess_data(df, methods=None, manual_vals=None):
    '''
    Preprocesses the data

    Inputs:
    df (pandas dataframe): the dataset
    methods (dict): keys are column names and values the imputation method to
        apply to that column; valid methods are defined in pipeline_library
    manual_vals (dict): keys are column names and values the values to fill
        missing values with in columns with 'manual' imputation method

    Returns: pandas dataframe
    '''
    df = pl.preprocess_data(df, methods=methods, manual_vals=manual_vals)

    return df

def generate_features(training, testing, n_ocurr_cols, scale_cols, bin_cols,
                      dummy_cols, binary_cut_cols, drop_cols):
    '''
    Generates categorical, binary, and scaled features. While features are
    generate for the training data independent of the testing data, features
    for the testing data sometimes require ranges or other information about the
    properties of features created for the training data to ensure consistency.
    Operations will occurr in the following order:
    - Create number of occurences columns, name of each column will be the name
      of the original column plus the suffix '_n_ocurr' (new column created)
    - Scale columns (replaces original column)
    - Bin columns (replaces original column)
    - Create dummy columns (replaces original column)
    - Binary cut columns (replaces original column)
    - Drop columns (eliminates original column)

    As such, number of occurence columns may be scaled, binned, etc, by
    specifying '<col_name>_n_ocurr' in the arguments. Binned columns will
    automatically be converted to dummies

    Inputs:
    training (pandas dataframe): the training data
    testing (pandas dataframe): the testing data
    n_ocurr_cols (list of strs): names of columns to count the number of
        ocurrences of each value for
    scale_cols (list of strs): names of columns to rescale to be between -1 and
        1
    bin_cols (dict): each key is the name of a column to bin and each value is a
        dictionary of arguments to pass to the cut_variable function in
        pipeline_library (must contain a value for bin (a binning rule), 
        labels and kwargs parameters are optional)
    dummy_cols (list of strs): names of columns to convert to dummy variables
    binary_cut_cols (dict of dicts): each key is the name of a column to cut
        into two groups based on some threshold and each value is a dictionry
        of arguments to pass to the cut_binary function in pipeline_library
        (must contain a value for threshold, or_equal_to parameter is optional)
    drop_cols (list of strs): names of columns to drop

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

    for col in scale_cols:
        max_training = max(training[col])
        min_training = min(training[col])
        training.loc[:, col] = pl.scale_variable_minmax(training[col])
        testing.loc[:, col] = pl.scale_variable_minmax(testing[col], a=max_training,
                                                b=min_training)

    for col, specs in bin_cols.items():
        training.loc[:, col], bin_edges = pl.cut_variable(training[col], **specs)
        bin_edges[0] = - float('inf') #test observations below the lowest observation
        #in the training set should be mapped to the lowest bin
        bin_edges[-1] = float('inf') #test observations above the highest observation
        #in the training set should be mapped to the highest bin
        testing[col], _ = pl.cut_variable(testing[col], bin_edges)

    dummy_cols += list(bin_cols.keys())
    for col in dummy_cols:
        values = list(training[col].value_counts().index)
        training = pl.create_dummies(training, col, values=values)
        testing = pl.create_dummies(testing, col, values=values)

    for col, specs in binary_cut_cols.items():
        training[col] = pl.cut_binary(training[col], **specs)
        testing[col] = pl.cut_binary(testing[col], **specs)

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
        features = training[i].drop('not_fully_funded_60days', axis=1)
        target = training[i].not_fully_funded_60days
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
        features = testing_splits[i].drop('not_fully_funded_60days', axis=1)
        pred_probs.append(pl.predict_target_probability(trained_classifiers[i],
                                                        features))

    return pred_probs

def evaluate_classifiers(pred_probs, testing_splits, seed=None, model_name=None,
                         fig_name=None):
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
    fig_name (str): prefix of file name to save the precision/recall curve in;
        if not specified the figure is displayed but not saved
    '''
    table = pd.DataFrame()
    for i in range(len(pred_probs)):
        print('Evaluating predictions with training set {}'.format(i+1))
        y_actual = testing_splits[i].not_fully_funded_60days
        table['Test/Training Set {}'.format(i + 1)], fig =\
            pl.evaluate_classifier(pred_probs[i], y_actual,\
            [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50], seed=seed, 
            model_name=model_name,
            dataset_name='Training/Testing Set # {}'.format(i + 1))
        if fig_name is not None:
            plt.save_fig(fig_name + '_set' + str(i) + '.png')
        else:
            plt.show()
    print(table)

def parse_args(args):
    '''
    Parses command line arguments for use by the rest of the software

    Inputs:
    args (argsparse Namespace): arguments from the command line

    Returns: 5-ple of filepath to dataset (str), pre-procesing specs (dict),
    feature generation specs (dict), model specs (list of dicts), seed (int)
    '''
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
