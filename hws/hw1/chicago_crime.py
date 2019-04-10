'''
Homework #1: Diagnostic
CAPP 30254: Machine Learning for Public Policy (Spring 2019)

Ben Fogarty
'''

from io import StringIO
from textwrap import wrap
from sodapy import Socrata
import geopandas as geopd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np
import seaborn as sns
import shapely
import urllib3


APP_TOKEN = ''
CENSUS_API = ''

def read_api_tokens(filename):
    '''
    Reads in API keys from a locally stored file. Within the filed, the Chicago
    Open Data Portal API key should be first, followed by the Census Bureau
    API key. Updats APP_TOKEN and CENSUS_API global variables.

    Inputs:
    filename (str): the location of the file to load api keys from
    '''
    with open(filename) as f:
        tokens = f.readlines()
    global APP_TOKEN
    APP_TOKEN = tokens[0].strip()

    global CENSUS_API
    CENSUS_API = tokens[1].strip()

def download_crime_reports(start_year, end_year):
    '''
    Imports crime reports data from the Chicago open data portal using the SODA
    API.

    Inputs:
    start_year (int): the first year to download crime reports from (valid input
        is 2001-2018)
    end_year (int): the last year to dowload crime reports from (valid input is
        2001-2018)

    Returns: pandas dataframe where each row is a crime report
    '''
    coltypes = {'latitude': float,
                'longitude': float,
                'year': int
                }
    client = Socrata('data.cityofchicago.org', APP_TOKEN)
    where_clause = 'year between {} and {}'.format(start_year, end_year)
    max_size = int(6.85 * 10 ** 6)
    results = client.get('6zsd-86xi', where=where_clause, limit=max_size)
    results_df = pd.DataFrame.from_records(results)\
                             .astype(coltypes)

    results_df.date = pd.to_datetime(results_df.date)

    return results_df

def download_community_areas():
    '''
    Imports names, numbers, and shapes of Chicago community areas from the
    Chicago Open Data Portal using the SODA API.

    Returns: geopandas geodataframe where each row is a community area
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
    crime_reports (pandas dataframe): each row is a crime report with a numeric
        column titled 'community_areas' that contains data on
    community_areas (geopandas geodataframe or pandas dataframe): each row is a
        community area with the area number in a column titled 'area_number' and
        the community name in a column title 'community'

    Returns: pandas dataframe where each row is a crime_report
    '''
    communities = pd.Series(index=community_areas.area_number.values,
                            data=community_areas.community.values)
    crime_reports['community_area'] = crime_reports.community_area\
                                                   .map(communities)\
                                                   .astype('category')
    return crime_reports

def download_blocks():
    '''
    Imports the names, numbers, and shapes of Chicago 2010 census blocks from
    the Chicago Open Data Portal using the SODA API.

    Returns: geopandas geodataframe where each row is a community area
    '''
    client = Socrata('data.cityofchicago.org', APP_TOKEN)
    max_size = 50000
    blocks = client.get('bt9m-d2mf', limit=max_size)
    blocks_df = pd.DataFrame.from_records(blocks)
    blocks_df['geoid10'] = blocks_df.geoid10.astype(str)
    #generates the geoid10 for the census block group
    blocks_df['geoid10'] = blocks_df.geoid10.apply(lambda x: x[:-3])
    blocks_df = blocks_df.loc[:, ['geoid10', 'the_geom']]

    blocks_df['the_geom'] = blocks_df.the_geom\
                                     .apply(shapely.geometry.shape)
    blocks_df = geopd.GeoDataFrame(blocks_df, geometry='the_geom')
    blocks_df.crs = {'init': 'epsg:4326'}

    return blocks_df

def link_reports_blocks(crime_reports):
    '''
    Performs a spatial join to link crime reports with the 2010 Census block in
    in which they occurred. Reports which cannot be linked are automatically
    dropped from the resulting geodataframe

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report

    Returns: geopandas geodataframe, where each row is a crime_report

    Citation:
    Creating geopandas geodata frame from dataframe with coordinates:
        http://geopandas.org/gallery/create_geopandas_from_pandas.html?
        highlight=zip
    '''
    blocks_df = download_blocks()

    crime_reports = crime_reports[crime_reports.longitude.notna() &
                                  crime_reports.latitude.notna()]\
                                 .copy() #shallow copy to avoid side effects
    crime_reports['the_geom'] = list(zip(crime_reports.longitude,
                                         crime_reports.latitude))
    crime_reports['the_geom'] = crime_reports.the_geom\
                                             .apply(shapely.geometry.Point)

    crime_reports = geopd.GeoDataFrame(crime_reports, geometry='the_geom')
    crime_reports.crs = {'init': 'epsg:4326'}

    crime_reports = geopd.sjoin(crime_reports, blocks_df, how='left',
                                op='within')
    crime_reports = crime_reports[crime_reports.geoid10.notna()].copy()

    return crime_reports

def get_block_stats():
    '''
    Downloads income, educational attainment, and race data from the 5-year ACS
    estimates for the block groups in the dataset

    Returns: pandas dataframe, where each row is a block group with income,
        educational attainment, and race data

    Citations:
    Making HTML Requests: https://urllib3.readthedocs.io/en/latest/
        user-guide.html
    Querying ACS Data: https://www.census.gov/content/dam/Census/data/
        developers/api-user-guide/api-guide.pdf
    Reading CSV from string: https://docs.python.org/2/library/stringio.html
    '''
    col_dict = {'B02001_001E': 'Race sample',
                'B02001_002E': 'White alone',
                'B02001_003E': 'Black alone',
                'B03003_001E': 'Hispanic or Latino sample',
                'B03003_003E': 'Hispanic or Latino',
                'B15003_001E': 'Education sample',
                'B15003_022E': "Bachelor's",
                'B15003_023E': "Master's",
                'B15003_024E': "Professional",
                'B15003_025E': 'PhD',
                'B19301_001E': 'Per Capita Income, last 12 months (2017 inflation adjusted dollars)'}
    query_address = ('http://api.census.gov/data/2017/acs/acs5?' +
                     'get={}&for=block%20group:*&in=state:17+' +
                     'county:031+tract:*&key={}')
    get_params = ",".join(list(col_dict.keys()))
    query_address = query_address.format('NAME,' + get_params, CENSUS_API)

    http = urllib3.PoolManager()
    urllib3.disable_warnings()
    request = http.request('GET', query_address)
    contents = request.data.decode('utf-8')
    contents = contents.replace('[', '')
    contents = contents.replace(']', '')

    col_types = {'B02001_001E': float,
                 'B02001_002E': float,
                 'B02001_003E': float,
                 'B03003_001E': float,
                 'B03003_003E': float,
                 'B15003_001E': float,
                 'B15003_022E': float,
                 'B15003_023E': float,
                 'B15003_024E': float,
                 'B15003_025E': float,
                 'B19301_001E': float,
                 'state': str,
                 'county': str,
                 'tract': str,
                 'block group': str}
    block_stats = pd.read_csv(StringIO(contents), dtype=col_types,
                              usecols=list(col_types.keys()))\
                    .rename(col_dict, axis=1)

    block_stats = transform_block_stats(block_stats)

    return block_stats

def transform_block_stats(block_stats):
    '''
    Transforms raws block-by-block statistics to the desired format

    Inputs:
    block_stats (pandas): each row is a block group and associated statistics

    Returns: pandas dataframe
    '''
    block_stats.copy() #shallow copy, avoids side effects
    block_stats = block_stats.replace({-666666666: float('nan')})

    block_stats['geoid10'] = block_stats.apply(lambda x: x.state + x.county +
                                               x.tract + x['block group'],
                                               axis=1)

    block_stats['White alone (%)'] = (block_stats['White alone'] /
                                      block_stats['Race sample'] * 100)
    block_stats['Black alone (%)'] = (block_stats['Black alone'] /
                                      block_stats['Race sample'] * 100)
    block_stats['Hispanic or Latino (%)'] = (block_stats['Hispanic or Latino'] /
                                             block_stats['Hispanic or Latino sample']
                                             * 100)

    block_stats["Bachelor's or more (>= 25 y/o) (%)"] = ((block_stats["Bachelor's"]
        + block_stats["Master's"] + block_stats['Professional'] +
        block_stats['PhD']) / block_stats['Education sample'] * 100)

    return block_stats

def link_reports_block_stats(crime_reports, block_stats):
    '''
    Links crime reports data with block group level statistics from the 5-year ACS

    Inputs:
    reports_df (pandas dataframe): each record contains a column 'geoid10' that
        denotes that records 2010 Census block group geoid
    block_stats (pandas dataframe): each row is a block groups with column
        'geoid10' denoting the row's 2010 Census block group geoid
        and associated statistics

    Returns: pandas dataframe
    '''
    crime_reports = pd.merge(crime_reports, block_stats, on='geoid10', how='left')
    return crime_reports

def summarize_on_field(crime_reports, summary_field):
    '''
    Summarizes a set of reported crimes by dividing it on some summary field

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    summary_field (str): field to summaries the data over, for example
        'primary_type'
    '''
    summarized = crime_reports[summary_field]\
                              .value_counts()\
                              .to_frame()\
                              .rename({'primary_type': 'Count'}, axis=1)

    summarized['Percentage'] = summarized.apply(lambda x: (x.Count /
                                                        np.sum(summarized.Count) * 100),
                                             axis=1)
    summarized.sort_values('Percentage', ascending=False)
    print(summarized)

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

    Citations:
    Creating month (text) labels: https://matplotlib.org/gallery/
        text_labels_and_annotations/date.html
    '''
    crime_reports = crime_reports.copy()
    f, ax = plt.subplots(nrows=1, ncols=1)
    crime_reports['month'] = crime_reports.date.dt.month\
                                          .apply(lambda x: '1900-{}-01'.format(x))
    crime_reports['month'] = pd.to_datetime(crime_reports.month)
    by_month = crime_reports.groupby(['month', 'year'])\
                            .count()\
                            .arrest\
                            .reset_index()\
                            .rename({'arrest': 'counts'}, axis=1)

    sns.lineplot(x='month', y='counts', hue='year', palette=['r', 'b'],
                 data=by_month, ax=ax)
    month_format = mdates.DateFormatter('%B')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(month_format)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Number of crime reports')
    ax.set_title('Number of crime reports by month')

    plt.subplots_adjust(bottom=0.15)
    plt.show()

def summarize_yoy_change(crime_reports, summary_field, start_year, end_year):
    '''
    Summarizes year to year changes in crime reports over some specified field

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    summary_field (str): field to summaries the data over, for example
        'primary_type'
    start_year: the base year
    end_year: the year to measue change to

    Citations:
    Wrapping long axis titles: https://stackoverflow.com/questions/15740682/
                               https://stackoverflow.com/questions/11244514/
    '''
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'hspace': 0.7})
    summarized = crime_reports.groupby(['year', summary_field])\
                                   .count()\
                                   .iloc[:, 0]\
                                   .unstack(level=0, fill_value=0)
    summarized['Change (absolute)'] = (summarized[end_year] -
                                       summarized[start_year])
    summarized.sort_values('Change (absolute)', inplace=True)
    print(summarized.to_string())

    #Decreases bar graph
    decreases = summarized.head(5)
    sns.barplot(x=list(decreases.index), y=decreases['Change (absolute)'],
                palette='Blues', ax=ax1)
    ax1.axhline()
    ax1.set_ylabel('Change in number of reports')
    title = "Largest absolute decrease between {} and {}"\
            .format(start_year, end_year)
    ax1.set_title(title)
    labels = ['\n'.join(wrap(l.get_text(), 10)) for l in ax1.get_xticklabels()]
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("")

    #Increases bar graph
    increases = summarized.tail(5)
    sns.barplot(x=list(increases.index), y=increases['Change (absolute)'],
                palette='Reds', ax=ax2)
    ax2.axhline()
    ax2.set_ylabel('Change in numbert of reports')
    title = "Largest absolute increase between {} and {}"\
            .format(start_year, end_year)
    ax2.set_title(title)
    labels = ['\n'.join(wrap(l.get_text(), 10)) for l in ax2.get_xticklabels()]
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("")

    plt.show()

def summarize_neighborhoods(crime_reports, community_areas):
    '''
    Summarizes the number of crime reports by neighborhood

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    community_areas (geopandas geodataframe): each row is a neighborhood
    '''
    by_neighborhood = crime_reports.community_area\
                                   .value_counts()\
                                   .sort_values()
    print(by_neighborhood.describe())
    print()

    graph_neighborhood_dist(crime_reports)
    print()

    map_neighborhood_stats(crime_reports, community_areas)
    print()

    print('Number of crime reports by neighborhood:')
    print(by_neighborhood)

def graph_neighborhood_dist(crime_reports):
    '''
    Generates a histogram showing the distribution of crime reports across
    different neighborhoods

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    '''
    by_neighborhood = crime_reports.community_area\
                                   .value_counts()\
                                   .sort_values()

    f, ax = plt.subplots(nrows=1, ncols=1)
    sns.set()
    sns.distplot(by_neighborhood, bins=range(0, 32001, 1000), kde=False, ax=ax)

    ax.set_xticks(range(0, 32001, 1000), minor=True)
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.set_xlabel('Number of crime reports')
    ax.set_ylabel('Number of neighborhoods')
    ax.set_title('Distribution of crime reports per neighborhood')

    plt.show()

def map_neighborhood_stats(crime_reports, community_areas):
    '''
    Generates a heatmap displaying the number of crimes reports across
    neighborhoods

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report
    community_areas (geopandas geodataframe): each row is a community area

    Citations:
    Heatmap: https://towardsdatascience.com/lets-make-a-map-using-
             geopandas-pandas-andg-matplotlib-to-make-a-chloropleth-map-
             dddc31c1983d
             (previously used in CS 122 Group Project)
    '''
    by_neighborhood = crime_reports.community_area\
                                   .value_counts()\
                                   .reset_index()\
                                   .rename({'community_area': 'counts',
                                            'index': 'community'}, axis=1)

    by_neighborhood = pd.merge(by_neighborhood, community_areas, on='community')
    by_neighborhood = geopd.GeoDataFrame(by_neighborhood, geometry='the_geom')
    by_neighborhood.crs = {'init': 'epsg:4326'}

    f, ax = plt.subplots(nrows=1, ncols=1)
    by_neighborhood.plot(column='counts', cmap='coolwarm', linewidth=0.8,
                         linestyle='-', ax=ax)

    ax.axis('off')
    scale_min = min(by_neighborhood.counts)
    scale_max = max(by_neighborhood.counts)
    legend = mpl.cm.ScalarMappable(cmap='coolwarm', norm=mpl.colors.Normalize(
                                   vmin=scale_min, vmax=scale_max))
    legend._A = []
    f.colorbar(legend, ax=ax)
    f.suptitle('Number of crime reports per neighborhood')

    plt.show()

def summarize_by_block(crime_reports, summary_field):
    '''
    Summarizes the profile of each block in a set of crime reports over a given
    summary field

    Inputs:
    crime_reports (pandas dataframe): each row is a crime report with a 'block'
        column giving the name of the block the crime occurred on and a
        'geoid10' giving the geoid of the block's 2010 Census block group
    summary_field (str): field to summarize the data over, for example
        'primary_type'

    Returns: pandas dataframe
    '''
    by_block = crime_reports.groupby(['block', 'geoid10', summary_field])\
                            .count()\
                            .iloc[:, 0]\
                            .unstack(level=2, fill_value=0)\
                            .reset_index()

    return by_block

def describe_blocks(block_summaries, block_stats):
    '''
    Describes the race, education, and income levels of blocks included in the
    block summaries dataset

    Inputs:
    block_summaries (pandas dataframe): each row is a summary of each block in
        a set of crime reports with a geoid10 column that contains the geoid
        of the block's 2010 Census block group
    block_stats (pandas dataframe): each row is demographic statistics on a
        block group and a geoid10 column containg the geoid of the block group's
        2010 Census block group
    '''
    linked = link_reports_block_stats(block_summaries, block_stats)
    cols_to_agg = ['White alone (%)', 'Black alone (%)',
                   'Hispanic or Latino (%)', "Bachelor's or more (>= 25 y/o) (%)",
                   'Per Capita Income, last 12 months (2017 inflation adjusted dollars)']

    return linked[cols_to_agg].describe()
