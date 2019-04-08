'''
Homework #1: Diagnostic
CAPP 30254: Machine Learning for Public Policy (Spring 2019)

Ben Fogarty
'''

from sodapy import Socrata
import pandas as pd
import geopandas as geopd
import shapely
import numpy as np

APP_TOKEN = '***REMOVED***'

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
    client = Socrata('data.cityofchicago.org', APP_TOKEN)
    where_clause = 'year between {} and {}'.format(start_year, end_year)
    max_size = int(6.85 * 10 ** 6)
    results = client.get('6zsd-86xi', where=where_clause, limit=max_size)
    results_df = pd.DataFrame.from_records(results)
    results_df.date = pd.to_datetime(results_df.date)

    return results_df

def download_community_areas():
    '''
    Imports the dataset containg the names, numbers, and shapes of Chicago
    community areas from the Chicago Open Data Portal using the SODA API.

    Returns: geopandas geodataframe of community areas
    '''
    client = Socrata('data.cityofchicago.org', APP_TOKEN)
    max_size = 100
    results = client.get('igwz-8jzy', limit=max_size)
    results_df = pd.DataFrame.from_records(results)
    results_df.rename({'area_numbe': 'area_number'}, axis=1, inplace=True)
    results_df['the_geom'] = results_df.the_geom\
                                       .apply(shapely.geometry.shape)
    results_df = geopd.GeoDataFrame(results_df, geometry='the_geom')
    results_df.crs = {'init': 'epsg:4326'}

    return results_df

def link_reports_neighborhoods(crime_reports, community_areas):
    '''
    Replaces the community areas number in a dataset of crime reports with the
    name of the community area

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    community_areas (geopandas geodataframe or pandas dataframe): each row is a
        community area with the area number in a column titled 'area_number' and
        the community name in a column title 'community'

    Returns: pandas dataframe of crime reports
    '''
    communities = pd.Series(index=community_areas.area_number,
                            data=community_areas.community)
    crime_reports['community_area'] = crime_reports.community_area\
                                                   .map(communities)\
                                                   .astype('category')
    return crime_reports

def summarize_crime_reports(crime_reports):
    '''
    Provides a general summary of a dataset of crime reports

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    #Overall
    n_reports = len(crime_reports)
    print('Overall, there were {} crime reports in the given set\n'.format(n_reports))

    #Summarize by type, if more than one time in the passed dataframe
    by_type = crime_reports.primary_type\
                           .value_counts()\
                           .to_frame()\
                           .rename({'primary_type': 'Count'}, axis=1)

    if len(by_type) > 1:
        by_type['Percentage'] = by_type.apply(lambda x: x.Count / np.sum(by_type.Count) * 100,
                                                     axis=1)
        by_type.sort_values('Percentage', ascending=False)
        print('Dividing based on types of crime:')
        print(by_type)
        print()

    ##Similar tactics for time, neighborhood, and arrest rate; refactor into
    ##separate function

##Two factor heat maps

