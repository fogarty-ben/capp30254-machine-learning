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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from textwrap import wrap

APP_TOKEN = 'rxYsI6aQTVNNqzshFFLTdecYL'

def download_crime_reports(start_year, end_year, community_areas):
    '''
    Imports the crime data from the Chicago open data portal using the SODA API.

    Inputs:
    start_year (int): the first year to download crime reports from (valid input
        is 2001-2018)
    end_year (int): the last year to dowload crime reports from (valid input is
        2001-2018)
    community_areas (geopandas geodataframe or pandas dataframe): each row is a
        community area with the area number in a column titled 'area_number' and
        the community name in a column title 'community'

    Returns: pandas dataframe of crime reports from the specified years
    '''
    client = Socrata('data.cityofchicago.org', APP_TOKEN)
    where_clause = 'year between {} and {}'.format(start_year, end_year)
    max_size = int(6.85 * 10 ** 6)
    results = client.get('6zsd-86xi', where=where_clause, limit=max_size)
    results_df = pd.DataFrame.from_records(results)
    
    results_df.date = pd.to_datetime(results_df.date)
    results_df = link_reports_neighborhoods(results_df, community_areas)

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
    communities = pd.Series(index=community_areas.area_number.values,
                            data=community_areas.community.values)
    print(communities)
    crime_reports['community_area'] = crime_reports.community_area\
                                                   .map(communities)\
                                                   .astype('category')
    return crime_reports

def summarize_types(crime_reports):
    '''
    Summarizes the types of crime in a set of reported crimes

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    by_type = crime_reports.primary_type\
                           .value_counts()\
                           .to_frame()\
                           .rename({'primary_type': 'Count'}, axis=1)

    by_type['Percentage'] = by_type.apply(lambda x: x.Count / np.sum(by_type.Count) * 100,
                                                 axis=1)
    by_type.sort_values('Percentage', ascending=False)
    print('Dividing based on types of crime:')
    print(by_type)

def summarize_yearly(crime_reports):
    '''
    Summarize year to year change in the number of crime reports

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    by_time = crime_reports.year\
                           .value_counts()\
                           .to_frame()\
                           .sort_index()\
                           .rename({'year': 'Number of Reports'}, axis=1)
    by_time['Percent Change'] = by_time['Number of Reports'].pct_change() * 100
    print(by_time)

def summarize_monthly(crime_reports):
    '''
    Summarizes month to month changes in the number of crime reports

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    f, ax = plt.subplots(nrows=1, ncols=1)
    crime_reports['month'] = crime_reports.date.dt.month\
                                          .apply(lambda x: '1900-{}-01'.format(x))
    crime_reports['month'] = pd.to_datetime(crime_reports.month)
    by_month = crime_reports.groupby(['month', 'year'])\
                            .count()\
                            .arrest\
                            .reset_index()\
                            .rename({'arrest': 'counts'}, axis=1)
    print(by_month)

    #https://matplotlib.org/gallery/text_labels_and_annotations/date.html
    sns.lineplot(x='month', y='counts', hue='year', palette=['r', 'b'],
                 data=by_month, ax=ax)
    month_format = mdates.DateFormatter('%B')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(month_format)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Number of crimes')
    ax.set_title('The number of crime reports appears to vary seasonally')
    
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def summarize_types_change(crime_reports, start_year, end_year):
    '''
    Summarizes year to year changes in types of crime reported

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    start_year: the base year
    end_year: the year to measue change to
    '''
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'hspace': 0.5})
    by_time_by_type = crime_reports.groupby(['year', 'primary_type'])\
                                   .count()\
                                   .arrest\
                                   .unstack(level=0)\
                                   .reset_index()
    by_time_by_type['change'] = (by_time_by_type[end_year] - 
                                 by_time_by_type[start_year])
    by_time_by_type.sort_values('change', inplace=True)
    print(by_time_by_type.loc[:, [start_year, end_year, 'change']])

    #Decreases bar graph
    sns.barplot(x='primary_type', y='change', data=by_time_by_type.head(5),
                palette='Blues', ax=ax1)
    ax1.axhline()
    ax1.set_ylabel('Change in number of reports')
    title = "Types of reports with largest absoulte decrease between {} and {}"\
            .format(start_year, end_year)
    ax1.set_title(title)
    labels = ['\n'.join(wrap(l.get_text(), 12)) for l in ax1.get_xticklabels()] #https://stackoverflow.com/questions/11244514/
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("")

    #Increases bar graph
    sns.barplot(x='primary_type', y='Change', data=by_time_by_type.tail(5),
                palette='Reds', ax=ax2) 
    ax2.axhline()
    ax2.set_ylabel('Change in numbert of reports')
    title = "Types of reports with largest absoulte increase between {} and {}"\
            .format(start_year, end_year)
    ax2.set_title(title)
    labels = ['\n'.join(wrap(l.get_text(), 12)) for l in ax2.get_xticklabels()]
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("")
    
    plt.show()

def summarize_neighborhoods(crime_reports, community_areas):
    '''
    Summarizes the number of crime reports by neighborhood

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

    disaggregated_arrest_rates(crime_reports, 'community_area')
    plt.show()

def map_neighborhood_stats(crime_reports, community_areas):
    '''
    Generates a heatmap displaying the number of crimes reports across
    neighborhoods

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    community_areas (geopandas geodataframe): each row is a community areas with
        column describing the geometry of that community area
    '''
    by_neighborhood = crime_reports.community_area\
                                   .value_counts()
    by_neighborhood = geopd.GeoDataFrame(community_areas.join(by_neighborhood))
    
    f, ax = plt.subplots(nrows=1, ncols=1)
    by_neighborhood.plot()





def summarize_crime_reports(crime_reports):
    '''
    Provides a general summary of a dataset of crime reports

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    #Overall
    n_reports = len(crime_reports)
    print('Overall, there were {} crime reports in the given set\n'.format(n_reports))
        

    ##Similar tactics for neighborhood, and arrest rate; add ability to filter

##Two factor heat maps?

