import pandas as pd
import missingno as msno

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

    #Display gaps (from missing values) in the dataset
    #msno.matrix(df)


    print('x')




if __name__ == '__main__':
    createStats()