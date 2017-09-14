import pandas as pd
import missingno as msno
import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib

from ReadData import readData

def createStats():

    #Read parsed dataframe
    df = readData()

    #Basic statistics will be conducted on the whole dataset (train + test)

    #Verify that all columns are of the appropriate datatypes
    dfTypes = df.dtypes
    #If not, assign data type
    df['#2 (detailed industry recode) nominal'] = df['#2 (detailed industry recode) nominal'].astype(str)
    df['#3 (detailed occupation recode) nominal'] = df['#3 (detailed occupation recode) nominal'].astype(str)
    df['#35 (own business or self employed) nominal'] = df['#35 (own business or self employed) nominal'].astype(str)
    df['#37 (veterans benefits) nominal'] = df['#37 (veterans benefits) nominal'].astype(str)
    df['#39 (year) nominal'] = df['#39 (year) nominal'].astype(str)

    #Display gaps (from missing values) in the dataset - remove comment for display
    #It is commented out as 'missingno' library automatically pauses script running to display graph
    #msno.matrix(df)
    #On a first parse, there appears to be no missing data

    #NOMINAL FEATURES
    #Store continuous variable names
    nominalVariableNames = []
    for columnName in df.columns:

        #Check if the variable is nominal - alternatively, could have checked by dtype
        if re.search('nominal', columnName):

            #Strip whitespace
            df[columnName] = df[columnName].str.strip()

            #Append to storage list for later processing - for example for one-hot encoding
            nominalVariableNames.append(columnName)

            #Plot counts of nominal values - commented out for simplicity of results
            '''plt.figure()
            sns.countplot(x=columnName, data=df)
            plt.xticks(rotation=30)'''

            #Print top/bottom 3 lables and information about them:
            labelCounts = df[columnName].value_counts()
            numberOfLabels = len(labelCounts)

            if len(labelCounts) <= 6:
                print('\nColumn "%s" has less than 7 labels (%s labels), all are printed: ' %
                      (columnName, numberOfLabels))
                print(labelCounts)
            else:
                print('\nColumn "%s" has 7 or more labels (%s labels), most and least frequent 3 are printed: ' %
                      (columnName, numberOfLabels))
                print(labelCounts[0:3])
                print(labelCounts[-3:])

            #Calculate % for most and least frequent labels
            labelSum = sum(labelCounts)
            mostFrequentPercent = labelCounts[0] / labelSum
            leastFrequentPercent = labelCounts[-1] / labelSum
            print('The most frequent label accounts for %s of the data, while the least frequent accounts for %s.' %
                  (mostFrequentPercent, leastFrequentPercent))

    #CONTINUOUS VARIABLES
    #Firstly, provide pandas summary table - provides data counts, mean, std, min, max, 25/50/75 percentiles
    print('\n', df.describe())
    #Shows that wage per hour / capital gains / capital losses / dividends from stock are mostly 0 - potentially missing

    #Plot correlations
    corr = df.corr()
    plt.figure()
    sns.heatmap(corr)
    plt.xticks(rotation=25)
    plt.yticks(rotation=0)
    plt.title('Correlations between continuous variables')
    #Strong correlations between weeks worked in year and number of persons worked for employer

    #Store continuous variable names
    continousVariableNames = []
    for columnName in df.columns:

        #Check if the variable is continuous - alternatively, could have checked by dtype
        if re.search('continuous', columnName):
            #Append to storage list for later processing - for example for hypothesis testing (not currently implemented)
            continousVariableNames.append(columnName)

            #Plot histograms / violin plots for each continuous variable to show distribution of values
            #Commented out for simplicity of results
            '''plt.figure()
            sns.distplot(df[columnName], rug=False, kde=False)
            plt.figure()
            sns.violinplot(df[columnName])'''
            #Histograms are dominated by one or two values (some which appear to be filled in for missing values)
                #or a large imbalance in the population



    #Some additional visualizations are created in respect to the target variable

    #We would like to understand the relationship between a few different features and the target:
    plt.figure()
    sns.countplot(x='#0 (age) continuous', hue='#40 (target) nominal', data=df)
    plt.xticks(rotation=30)
    plt.figure()
    sns.countplot(x='#4 (education) nominal', hue='#40 (target) nominal', data=df)
    plt.xticks(rotation=30)
    plt.figure()
    sns.countplot(x='#8 (major industry code) nominal', hue='#40 (target) nominal', data=df)
    plt.xticks(rotation=30)
    plt.figure()
    sns.countplot(x='#34 (citizenship) nominal', hue='#40 (target) nominal', data=df)
    plt.xticks(rotation=30)
    plt.show()

    #A few interesting insights come forward, such as:
        #In terms of education, the ratio of people earning less than than 50k vs more than 50k
            #significantly decreases with the advancement in degree. However, this does not hold for Masters degrees,
            #and needs further investigation. For example, it might either be true as is, it might be an after-effect
            #of the sampling procedure employed by the census, it might be due to the fact that most of the
            #well-trained Masters students go on to do a PhD, etc.



    #Based on the information observed in the dataframe, missing values appear to be replaced with
    #'Not in universe' or 0 or '0' or '?'
    dfNullsIncluded = df.replace(['Not in universe', 0, '0', '?', 'Not in universe or children', 'Nonfiler', ],
                                 value=np.nan)
    #Replacing the above with null values, we can create the missing value chart again
    #msno.matrix(dfNullsIncluded)

    #Based on this missing value matrix, we recommend data improvements across the board
    #Due to time limitations for this exercise, the analysis procedure ends here
    #However, we would spend additional time understanding the dataset, its source, and provide a highly
        #detailed report for each variable, or focus on a particular area, depending on task specifications

    #Dump the transformed dataset (and column names) into a pickle file for easy pickup in later scripts
    joblib.dump(df, 'J:\Datasets\Exercises\Exercise5\BasicDataset.pkl')
    joblib.dump((nominalVariableNames, continousVariableNames), 'J:\Datasets\Exercises\Exercise5\VariableNames.pkl')