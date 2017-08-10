import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def main():

    #Read in the two datasets as dataframes
    dfRevenue = pd.read_csv('dataset2.csv')
    dfPromo = pd.read_csv('dataset2b.csv')

    #Calculate daily revenue & daily number of players
    dailyRevenue = dfRevenue.groupby('Date')['Revenue'].sum()
    dailyPlayers = dfRevenue.groupby('Date')['Playerid'].count()
    dailyPlayers.rename('Players', inplace=True)

    #Join the two series
    df = pd.concat([dailyRevenue, dailyPlayers], axis=1).reset_index()

    #Convert Date column to datetime format and sort df
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values('Date', inplace=True)

    #Get weekday - Monday=0, Sunday=6
    df['Weekday'] = df['Date'].dt.dayofweek

    #Add the type of promotion on the day
    dfPromo['Date'] = pd.to_datetime(dfPromo['Date'], dayfirst=True)
    df = df.merge(dfPromo, how='left', on='Date')


    #Check dataframe for missing values
    dfDescribe = df.describe()
    #Fill promo missing values
    df['Promo'].fillna('None', inplace=True)
    #Encode promotion into 3 separate columns for modeling purposes
    dfPromoEncoded = pd.get_dummies(df['Promo'], prefix='Promo')
    #Merge promo onto df and remove old Promo column
    df = df.merge(dfPromoEncoded, how='left', left_index=True, right_index=True)
    df.drop('Promo', inplace=True, axis=1)

    #Repeat encoding procedure for weekdays
    dfWeekdayEncoded = pd.get_dummies(df['Weekday'], prefix='Weekday')
    df = df.merge(dfWeekdayEncoded, how='left', left_index=True, right_index=True)
    df.drop('Weekday', inplace=True, axis=1)

    #Separate features and target variable from the dataset
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1]

    #Standardize features - not an improvement in this particular case
    #sc = StandardScaler()
    #X = sc.fit_transform(X)

    #Create model
    lr = LinearRegression()
    #Fit model and get residuals, perform cross validation
    lrFit = lr.fit(X, y)
    y_Fitted = lrFit.predict(X)
    residuals = y - y_Fitted
    CV = cross_val_predict(lr, X, y, cv=10, verbose=1)
    #Compare different scoring methods
    score = metrics.explained_variance_score(y, CV)
    score2 = metrics.mean_absolute_error(y, CV)
    score3 = metrics.mean_squared_error(y, CV)
    score4 = metrics.median_absolute_error(y, CV)
    score5 = metrics.r2_score(y, CV)
    print('Mean Absolute Error = %s \n Rsq = %s' % (score2, score5))

    #Create prediction ddictionary and put into dataframe
    testData = {'Players': [3000, 4000, 4000, 5000, 6000, 6000, 7000],
                'Promo': ['None', 'A', 'None', 'B', 'A', 'B', 'None'],
                'Weekday': [0, 1, 2, 3, 4, 5, 6]}
    dfTest = pd.DataFrame(testData)
    #Encode Promo and Weekday
    dfPromoEncoded = pd.get_dummies(dfTest['Promo'], prefix='Promo')
    dfTest = dfTest.merge(dfPromoEncoded, how='left', left_index=True, right_index=True)
    dfTest.drop('Promo', inplace=True, axis=1)
    dfWeekdayEncoded = pd.get_dummies(dfTest['Weekday'], prefix='Weekday')
    dfTest = dfTest.merge(dfWeekdayEncoded, how='left', left_index=True, right_index=True)
    dfTest.drop('Weekday', inplace=True, axis=1)

    #Predict revenue
    X_test = dfTest.iloc[:, :]
    y_test = lrFit.predict(X_test)

    #Insert results into test dataframe and compute 95% CI
    dfTest['Revenue'] = y_test
    dfTest['LowerCI'] = dfTest['Revenue'] - 1.96 * df['Revenue'].std() / math.sqrt(len(df))
    dfTest['UpperCI'] = dfTest['Revenue'] + 1.96 * df['Revenue'].std() / math.sqrt(len(df))

    #Plot residuals
    plt.figure(figsize=(16, 10))
    residuals.index = df['Date']
    plt.scatter(residuals.index, residuals)
    ax = plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals')
    plt.title('Linear Regression Residuals Plot')
    plt.show()

    #a) Predicted revenue is calculated in lines 88-89
    #b) Based purely on the linear model found, we can conclude that Promotion B performs better.
        #This is based on the LR coefficients of each of the promotion types:
        #Promo A: -342; Promo B: 2307; Promo None: -1964
        #Further correlation analysis can be used to confirm these results
    #c) Confidence intervals are calculated in lines 92-94
    #d) Residuals are plotted in lines 97-104
        #The residuals are randomly scattered around the zero horizontal axis.
        #This is a strong indicator that a linear model is a good fit for the dataset
        #The residual values are mostly within the MAE (+/- 1500)
        #There are outliers that warrant further investigation
    #e) The type of problem which predicts revenue based on number of players as well as supporting information
        #from the organization can be further developed. It would involve more steps, from understanding what
        #data is available, what other features could we use (and if they are not collected and stored, implement
        #procedures to do so), to different modeling techniques depending on the performance of linear or
        #non-linear regression models.


if __name__ == '__main__':
    main()