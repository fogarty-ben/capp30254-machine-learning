'''
Simple Machine Learning Pipeline

Ben Fogarty

18 April 2018
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from textwrap import wrap

def read_csv(filepath, cols=None, col_types=None):
	'''
	Imports a CSV file into a pandas data frame, optionally specifying columns
	to import and the types of the columns.

	Inputs:
	filepath (str): the path to the file
	row_index (str or list of strs): an optional column name or list of column
		names to index the rows of the dataframe with
	cols (list of strings): an optional list of columns from the data to import;
		only to be used if the first line of the csv is a header row
	col_types (dict mapping strings to types): an optional dictionary specifying
		the data types for each column; each key must be a the name of a column
		in the dataset and the associate value must be a pandas datatype (valid
		types listed here: http://pandas.pydata.org/pandas-docs/stable/
		getting_started/basics.html#dtypes)

	Returns: pandas dataframe
	'''
	return pd.read_csv(filepath, usecols=cols, dtype=col_types)

def show_distribution(df, variable):
	'''
	Graphs a histogram and the approximate distribtion of one variable in a
	dataframe.

	Inputs:
	df (pandas dataframe): dataframe containing the variable to show the
		distribution of as a column
	variable (str): the variable to show the distribution of; must be the name
		of a column in the dataframe

	Returns: matplotlib figure

	Citations:
	Locating is_numeric_dtype: https://stackoverflow.com/questions/19900202/
	'''
	sns.set()
	if pd.api.types.is_numeric_dtype(df[variable]):
		f, (ax1, ax2) = plt.subplots(2, 1)
		sns.distplot(df[variable], kde=False, ax=ax1)
		sns.distplot(df[variable], hist=False, kde_kws={'shade': True},
					 rug=True, ax=ax2)
		ax1.set_title('Histogram')
		ax1.set_ylabel('Count')
		ax2.set_title('Estimated density')
	else:
		f, ax = plt.subplots(1, 1)
		val_counts = df[variable].value_counts()
		sns.barplot(x=val_counts.index, y=val_counts.values, ax=ax)
		ax.set_ylabel('Count')
		
	f.suptitle('Distribution of {}'.format(variable))
	f.subplots_adjust(hspace=.5, wspace=.5)

	return f

def pw_correlate(df, variables, visualize=False):
	'''
	Calculates a table of pairwise correlations between numberic variables.

	Inputs:
	df (pandas dataframe): dataframe containing the variables to calculate
		pairwise correlation between
	variables (list of strs): the variables to calculate pairwise correlations
		between; each passed str must be name of a numeric type (including 
		booleans) column in the dataframe
	visualize (bool): optional parameter, if enabled the function prints out
		a heat map to help draw attention to larger correlation coefficients 

	Returns: pandas dataframe

	Wrapping long axis labels: https://stackoverflow.com/questions/15740682/
                               https://stackoverflow.com/questions/11244514/
	'''
	corr_table = np.corrcoef(df.loc[:, variables].dropna(), rowvar=False)
	corr_table = pd.DataFrame(corr_table, index=variables, columns=variables)
	
	if visualize:
		sns.set()
		f, ax = plt.subplots(figsize=(8, 6))
		sns.heatmap(corr_table, annot=True, fmt='.3f', linewidths=0.5, vmin=0,
					vmax=1, square=True, cmap='coolwarm', ax=ax)
		labels = ['-\n'.join(wrap(l.get_text(), 12)) for l in ax.get_yticklabels()]
		ax.set_yticklabels(labels)
		ax.tick_params(axis='both', rotation=0, labelsize='small')
		f.suptitle('Correlation Table')
		f.subplots_adjust(left=0.25)
		f.tight_layout()
		f.show()

	return corr_table