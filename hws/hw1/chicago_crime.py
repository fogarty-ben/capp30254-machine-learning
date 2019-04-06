'''
Homework #1: Diagnostic
CAPP 30254: Machine Learning for Public Policy (Spring 2019)

Ben Fogarty
'''

import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
import seaborn as sns

def go():
    '''
    Runs everything needed for the analysis
    '''
    #Problem 1
    crime_reports = download_crime_reports(2017, 2018)
    summarize_crime(crime_reports)

def download_crime_reports(start_year, end_year):
    '''
    Imports the crime data from the Chicago open data portal using the SODA API.

    Inputs:
    start_year (int): the first year to download crime reports from (valid input
        is 2001-2018)
    end_year (int): the last year to dowload crime reports from (valid input is
        2001-2018)

    Returns: pandas dataframe of crime reports from the specified years
    '''
    client = Socrata('data.cityofchicago.org', 'rxYsI6aQTVNNqzshFFLTdecYL')
    where_clause = 'year between {} and {}'.format(start_year, end_year)
    max_size = int(6.85 * 10 ** 6)
    results = client.get('6zsd-86xi', where=where_clause, limit=max_size)
    results_df = pd.DataFrame.from_records(results)

    return results_df

def summarize_crime_reports(crime_reports):
    '''
    Generates summary statistics of crime reports data

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    '''
    Too detailed types
    groupby = crime_reports.groupby(['primary_type', 'description'])
    by_type = pd.DataFrame(columns=['Type', 'Description', 'Count'])
    for (type_, desc), group in groupby:
        by_type = by_type.append({'Type': type_, 'Description': desc, 'Count':
                        len(group)}, ignore_index=True)
    print(by_type.sort_values('Count', ascending=False).head(10))
    '''
    crime_reports.primary_type.value_counts().plot(kind='pie')
    plt.show()
