'''
Homework #1: Diagnostic
CAPP 30254: Machine Learning for Public Policy (Spring 2019)

Ben Fogarty
'''

import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def go():
    '''
    Runs everything needed for the analysis
    '''
    #Problem 1
    crime_reports = download_crime_reports(2017, 2018)
    print('In total, there were {} crime reports in Chicago between {} and {}'\
          .format(len(crime_reports), 2017, 2018))
 
    summarize_by_type(crime_reports)

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

def summarize_by_time(crime_reports):
    '''
    Generates summary statistics of crime reports data

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
   #By type of crime
    by_type = crime_reports.primary_type\
                           .value_counts()\
                           .to_frame()\
                           .rename({'primary_type': 'Count'}, axis=1)
    by_type.loc[:, 'Percentage'] = by_type.apply(lambda x: x.Count / np.sum(by_type.Count) * 100,
                                                 axis=1)\
                                          .rename({'0': 'Percentage'}) 
    print(by_type)
    top10 = by_type.Count.head(5).copy()
    top10['Other types'] = np.sum(by_type.Count.tail(len(by_type) - 5))
    plt.pie(top10, autopct='%1.1f%%', labels=list(top10.index)) 
    plt.title('Five types of crime account for 68% of all crime reports')
    plt.show()
   
def summarize_by_time(crime_reports):
    '''
    Temportal analysis of crime reports data

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    pass 
