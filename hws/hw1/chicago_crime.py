'''
Homework #1: Diagnostic
CAPP 30254: Machine Learning for Public Policy (Spring 2019)

Ben Fogarty
'''

import pandas as pd
from sodapy import Socrata

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
    client = Socrata('data.cityofchicago.org', '***REMOVED***')
    where_clause = 'year between {} and {}'.format(start_year, end_year)
    results = client.get('6zsd-86xi', where=where_clause)
    results_df = pd.DataFrame.from_records(results)

    return results_df
