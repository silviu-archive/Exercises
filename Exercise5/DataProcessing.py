import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.externals import joblib

from BasicStatistics import createStats

def preprocessData():

    #If the 'BasicStatistics.py' script was never run (BasicDataset.pkl does not exist), run it
        #If clause not implemented
    #createStats()

    #Load information (learn/test datasets, variable names
    df = joblib.load('J:\Datasets\Exercises\Exercise5\BasicDataset.pkl')
    variableNames = joblib.load('J:\Datasets\Exercises\Exercise5\VariableNames.pkl')
    nominalVariableNames = variableNames[0]
    continousVariableNames = variableNames[1]

    #In regard to the continuous columns, there is not much score to create composite values
    #These would normally be used when trying to understand how features evolved since a particular point in time
    #e.g. since account creation, since time of use, etc. (if longstanding users are different to new users)
    
    #LATER EDIT: Further study of the dataset could result in the creation of the following composite columns
        #Wage per hour compared to people with the same education levels - for example with $x above/below average
            #similarly, for the same industry / occupation / class of worker / other nominal categories

    #We will however one-hot encode all the categorical variables for modelling purposes (except target)
        #(so as not to be limited to only models that can handle categorical features)
    df = pd.get_dummies(df, prefix=nominalVariableNames[:-1], columns=nominalVariableNames[:-1])

    #Move target to the start of the dataframe
    cols = list(df)
    #Move the target column to head of list using index, pop and insert
    cols.insert(0, cols.pop(cols.index('#40 (target) nominal')))
    #Use ix (mixed integer and label based access) to reorder
    df = df.ix[:, cols]

    return df
