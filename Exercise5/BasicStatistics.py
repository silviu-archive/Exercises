import pandas as pd
import missingno as msno
import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ReadData import readData


def createStats():

    #Read parsed dataframe
    df = readData()

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

    '''
    #NOMINAL FEATURES
    for columnName in df.columns:

        #Check if the variable is nominal - alternatively, could have checked by dtype
        if re.search('nominal', columnName):

            #Plot counts of nominal values
            plt.figure()
            sns.countplot(x=columnName, data=df)
            plt.xticks(rotation=30)

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
    '''

    #CONTINUOUS VARIABLES
    #Firstly, provide pandas summary table - provides data counts, mean, std, min, max, 25/50/75 percentiles
    print('\n', df.describe())

    #Plot correlations
    corr = df.corr()
    plt.figure()
    sns.heatmap(corr)
    plt.xticks(rotation=25)
    plt.yticks(rotation=0)
    plt.title('Correlations between continuous variables')

    continousVariableNames = []
    for columnName in df.columns:

        #Check if the variable is nominal - alternatively, could have checked by dtype
        if re.search('continuous', columnName):


            continousVariableNames.append(columnName)

            #Plot histograms
            plt.figure()
            sns.distplot(df[columnName], rug=False, kde=False)

    plt.show()

    print('x')




if __name__ == '__main__':
    createStats()