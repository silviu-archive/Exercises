import pandas as pd
import missingno as msno

def readData():

    #Define paths for learning and metadata files
    learnDatasetPath = 'J:\Datasets\Exercises\Exercise5\census_income_learn.csv'
    testDatasetPath = 'J:\Datasets\Exercises\Exercise5\census_income_test.csv'
    metadataPath = 'J:\Datasets\Exercises\Exercise5\census_income_metadata.txt'

    #Read metadata file line by line
    with open(metadataPath) as f:
        lines = f.readlines()

    #Parse lines in metadata file where column names are located
    #Parsing is done on specific hardcoded lines where confirmation exists on the data
    parsedLines = []
    for line in lines[81:121]:
        splitLine = line.split(sep='#')
        parsedLines.append('#' + splitLine[-1][:-1])
    #Append a target column name to the parsed lines
    parsedLines.append('#40 (target) nominal')

    #Read learning & test dataframe
    df = pd.read_csv(learnDatasetPath, header=None)
    dfTest = pd.read_csv(testDatasetPath, header=None)

    #Join learn and test together, but add a flag to be able to identify them later on
    df['TestSet'] = 0
    dfTest['TestSet'] = 1
    df = df.append(dfTest)

    #Drop 'instance weight' column as described in metadata
    df.drop([24], axis=1, inplace=True)
    #Assign column names from parser (including newly created 'TestSet' column)
    parsedLines.append('TestSet')
    df.columns = parsedLines

    return df