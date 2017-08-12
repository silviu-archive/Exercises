import pandas as pd
from sklearn.externals import joblib
import numpy as np

def createDataset():

    # Read dataset - tab separated, ignore bad lines
    df = pd.read_csv('J:/Datasets/viewership_extract.csv', error_bad_lines=False, sep="\t")
    print(df.head())

    # Create calculated columns to help with the analysis
    # Session length
    df['session_end'] = pd.to_datetime(df['session_end'])
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_length'] = df['session_end'] - df['session_start']
    # Broadcast length
    df['original_broadcast_end'] = pd.to_datetime(df['original_broadcast_end'])
    df['original_broadcast_start'] = pd.to_datetime(df['original_broadcast_start'])
    df['broadcast_length'] = df['original_broadcast_end'] - df['original_broadcast_start']
    # Age at the time of viewing
    df['dob'] = pd.to_datetime(df['dob']).dt.date
    df['age'] = (df['session_start'].dt.date - df['dob']).astype('<m8[Y]')
    # Bin ages
    bins = [0, 25, 35, 45, 55, 65, 115]
    df['age_bins'] = pd.cut(df['age'], bins=bins, include_lowest=False)
    # Transform length from timedelta to floats
    df['sessionLengthSeconds'] = df['session_length'].dt.seconds
    df['sessionLengthHours'] = df['sessionLengthSeconds'] / 3600
    df['broadcastLengthSeconds'] = df['broadcast_length'].dt.seconds
    df['broadcastLengthHours'] = df['broadcastLengthSeconds'] / 3600
    # Difference between broadcast and session
    df['startDifference'] = (df['session_start'] - df['original_broadcast_start']).dt.seconds
    df['sessionStartHour'] = df['session_start'].dt.hour
    df['sessionStartDay'] = df['session_start'].dt.dayofweek
    df['broadcastStartHour'] = df['original_broadcast_start'].dt.hour
    df['broadcastStartDay'] = df['original_broadcast_start'].dt.dayofweek

    #Save transformed dataset
    #df.to_csv('TransformedData.csv')
    #Trades off storage space vs load speed
    joblib.dump(df, 'J:/Source/Exercises/Exercise2/TransformedData.pkl')

if __name__ == '__main__':
    createDataset()