'''
Homework #1: Diagnostic
CAPP 30254: Machine Learning for Public Policy (Spring 2019)

Ben Fogarty
'''

import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    client = Socrata('data.cityofchicago.org', '***REMOVED***')
    where_clause = 'year between {} and {}'.format(start_year, end_year)
    max_size = int(6.85 * 10 ** 6)
    results = client.get('6zsd-86xi', where=where_clause, limit=max_size)
    results_df = pd.DataFrame.from_records(results)
    results_df.date = pd.to_datetime(results_df.date)
    return results_df

def summarize_by_type(crime_reports):
    '''
    Summarizes crime reports data by type of crime

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

    f2, ax2 = plt.subplots(nrows=1, ncols=1)
    sns.distplot(arrest_by_neigh, bins=range(0, 39, 2), kde=False, ax=ax2)
    ax.set_xticks(range(0, 39, 2), minor=True)
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax2.set_xlabel('Perecent of reports ending in an arrest')
    ax2.set_ylabel('Number of neighborhoods')
    ax2.set_title('The percent of reports resulting in an arrest is more even' +
    ' across neighborhoods, though some outliers remain', wrap=True)



    plt.show()
   
def summarize_by_time(crime_reports):
    '''
    Temportal analysis of crime reports data

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True,
    gridspec_kw={'hspace': 0.44444})
   
    by_time = crime_reports.year\
                           .value_counts()
    by_time['Pct Change'] = ((by_time['2018'] - by_time['2017']) /
                              by_time['2017'] * 100)
    print(by_time)

    by_time_by_type = crime_reports.groupby(['year', 'primary_type'])\
                                   .count()\
                                   .arrest\
                                   .unstack(level=0)\
                                   .reset_index()
    by_time_by_type['change'] = ((by_time_by_type['2018'] - by_time_by_type['2017'])
                                  / by_time_by_type['2017'] * 100)
    by_time_by_type.sort_values('change', inplace=True)
    print(by_time_by_type)

    sns.barplot(x='primary_type', y='change', data=by_time_by_type.head(5),
                palette='Blues', ax=ax1)
    ax1.axhline()
    ax1.set_ylabel('Percente change')
    ax1.set_title('Types of reports with the biggest percentage decrease between 2017 and 2018')

    sns.barplot(x='primary_type', y='change', data=by_time_by_type.tail(5),
                palette='Reds', ax=ax2) 
    ax2.axhline()
    ax2.set_ylabel('Percent change')
    ax2.set_title('Types of reports with the biggest percentage increase between 2017 and 2018')
    plt.xticks(wrap=True) 
    plt.show()

def summarize_by_month(crime_reports):
    '''
    Temportal analysis of crime reports data

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    f, ax = plt.subplots(nrows=1, ncols=1)
    crime_reports['month'] = crime_reports.date.dt.month
    by_month = crime_reports.groupby(['month', 'year'])\
                            .count()\
                            .arrest\
                            .reset_index()\
                            .rename({'arrest': 'counts'}, axis=1)

    print(by_month)
    sns.lineplot(x='month', y='counts', hue='year', palette=['r', 'b'],
                 data=by_month, ax=ax)
    ax.set_xlabel('Month of the year')
    ax.set_ylabel('Number of crimes')
    ax.set_xticks(range(13))
    ax.set_title('The number of crime reports appears to vary seasonally')
    plt.show()

def summarize_by_neighborhood(crime_reports):
    '''
    Summarizes crime reports data by neighborhood 

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    
    by_neighborhood = crime_reports.community_area\
                                   .value_counts()\
                                   .sort_values()
    print('Number of crimes:')
    print(by_neighborhood.agg([np.mean, np.std]))
    print(by_neighborhood.quantile([0, .25, .5, .75, 1])) 
    print()

    f, ax = plt.subplots(nrows=1, ncols=1)

    sns.distplot(by_neighborhood, bins=range(0, 32001, 1000), kde=False, ax=ax)
    ax.set_xticks(range(0, 32001, 1000), minor=True)
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.set_xlabel('Number of crime reports')
    ax.set_ylabel('Number of neighborhoods')
    ax.set_title('The distribution of crime reports per neighborhood has a long'
    + ' right tail')

   crime_reports['arrest'] = crime_reports.arrest.astype(int)
    arrest_by_neigh = crime_reports.groupby(['community_area'])\
                                   .arrest\
                                   .mean()
    arrest_by_neigh = arrest_by_neigh * 100
    print('Percent of reports with arrest')
    print(arrest_by_neigh.agg([np.mean, np.std]))
    print(arrest_by_neigh.quantile([0, .25, .5, .75, 1]))
    
    f2, ax2 = plt.subplots(nrows=1, ncols=1)
    sns.distplot(arrest_by_neigh, bins=range(0, 39, 2), kde=False, ax=ax2)
    ax.set_xticks(range(0, 39, 2), minor=True)
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax2.set_xlabel('Perecent of reports ending in an arrest')
    ax2.set_ylabel('Number of neighborhoods')
    ax2.set_title('The percent of reports resulting in an arrest is more even' +
    ' across neighborhoods, though some outliers remain', wrap=True)

    plt.show()

    
