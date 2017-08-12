import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import datetime
from calendar import isleap
import missingno as msno
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import FormatStrFormatter
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD, NMF, FastICA
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, VarianceThreshold
from sklearn.pipeline import FeatureUnion


def createModelingDataset():
    # Read transformed dataframe
    df = joblib.load('J:/Source/Exercises/Exercise2/TransformedData.pkl')

    # Transform dataset
    modelDf = pd.DataFrame()
    modelDf['ageBinTarget'] = df['age_bins'].astype(str)
    # Target exists in about 35% of variables
    modelDf['ID'] = df['household_id']
    modelDf['sessionStartHour'] = df['session_start'].dt.hour
    modelDf['sessionStartDayOfWeek'] = df['session_start'].dt.dayofweek
    modelDf['sessionEndHour'] = df['session_end'].dt.hour
    modelDf['sessionEndDayOfWeek'] = df['session_end'].dt.dayofweek
    modelDf['channelName'] = df['channel_name']
    modelDf['title'] = df['title']
    modelDf['broadcastStartHour'] = df['original_broadcast_start'].dt.hour
    modelDf['broadcastStartDay'] = df['original_broadcast_start'].dt.dayofweek
    modelDf['broadcastEndHour'] = df['original_broadcast_end'].dt.hour
    modelDf['broadcastEndDay'] = df['original_broadcast_end'].dt.dayofweek
    modelDf['sessionType'] = df['session_type']
    modelDf['sessionSubType'] = df['session_sub_type']
    modelDf['genre'] = df['genre']
    modelDf['subGenre'] = df['sub_genre']
    modelDf['playbackSpeed'] = df['playback_speed']
    modelDf['episodeTitle'] = df['episode_title']
    modelDf['seriesTitle'] = df['series_title']
    modelDf['gender'] = df['gender']
    modelDf['sessionLength'] = df['sessionLengthSeconds']
    modelDf['broadcastLength'] = df['broadcastLengthSeconds']
    modelDf['viewingDifference'] = df['startDifference']

    joblib.dump(modelDf, 'J:/Source/Exercises/Exercise2/ModelingData.pkl')

def main():

    # Read transformed dataframe
    df = joblib.load('J:/Source/Exercises/Exercise2/ModelingData.pkl')

    # Under normal circumstances the first step I would undertake on the transformed data would be to encode
    # Encoding can be done either through sklearn's one hot encoding, or through pandas get_dummies, as below:
    '''pd.get_dummies(df, prefix=['channelName', 'title', 'sessionType', 'sessionSubType', 'genre', 'subGenre',
                                    'episodeTitle', 'seriesTitle', 'gender'],
                   columns=['channelName', 'title', 'sessionType', 'sessionSubType', 'genre', 'subGenre',
                                    'episodeTitle', 'seriesTitle', 'gender'], sparse=True)'''
    # Unfortunately, even while using sparse matrices, the memory requirements exceed my current machine's capabilities



    #Due to hardware limitations, we need to come up with alternative solutions
    #We still need to aggregate categorical labels per household, but first let's reduce the dataset

    #Remove sessions with a 0 or negative length
    df = df.loc[df['sessionLength'] > 0]
    #Remove surf, due to our previous assumption that it is not actually a person watching a title
    df = df.loc[df['title'] != 'Surf']
    #Only look at normal playback speed
    df = df.loc[df['playbackSpeed'] == 1000]
    #Remove sessions with broadcast length < 0
    df = df.loc[df['broadcastLength'] > 0]
    #Do not consider sessions shorter than 15 seconds
    df = df.loc[df['sessionLength'] >= 15]

    #While there are concerns with removing the above pieces of information, we approach encoding again.
    #Unfortunately, our machine can still not handle the amount of data
    #Thus, we have 2 options:
    #1 - we resort to sampling procedures
    #2 - we create metrics based off of medians, quartiles, averages, etc. for each household


    #We decide to move forward with the sampling procedure
    #First we only select the data that has a target
    dfTrain = df.loc[df['ageBinTarget'] != 'nan']

    #Delete households with less than 10 views
    counts = dfTrain['ID'].value_counts()
    dfTrain = dfTrain[df['ID'].isin(counts[counts >= 10].index)]


    #Retrieve top 10 shows for each household
    store = pd.DataFrame()
    grouped = dfTrain.groupby('ID')
    for name, group in grouped:
        temp = group.title.value_counts().iloc[:5].reset_index()
        topTitles = temp.T.iloc[0, :]
        topTitlesCount = temp.T.iloc[1, :]
        combined = topTitlesCount.append(topTitles).reset_index(drop=True)
        combined.rename(name, inplace=True)
        store = store.append(combined)




    #Testing encoding on a 15% sample at this point still does not manage to provide results due to spec limits
    #Delete some more columns:
    del dfTrain['sessionSubType']
    del dfTrain['subGenre']
    del dfTrain['episodeTitle']
    del dfTrain['seriesTitle']
    del dfTrain['title']
    del dfTrain['playbackSpeed']
    del dfTrain['broadcastLength']

    #Removal of all the columns make the sampling procedure irrelevant, hence we 'sample' all the data
    sample = dfTrain.sample(frac=1, random_state=333)
    dummySample = pd.get_dummies(sample, prefix=['sessionStartHour', 'sessionStartDayOfWeek', 'sessionEndHour',
                                                 'sessionEndDayOfWeek','broadcastStartHour', 'broadcastStartDay',
                                                 'broadcastEndHour', 'broadcastEndDay', 'channelName',
                                                 'sessionType', 'genre', 'gender'],
                   columns=['sessionStartHour', 'sessionStartDayOfWeek', 'sessionEndHour',
                                                 'sessionEndDayOfWeek','broadcastStartHour', 'broadcastStartDay',
                                                 'broadcastEndHour', 'broadcastEndDay', 'channelName',
                                                 'sessionType', 'genre', 'gender'], sparse=True)
    #Groupby household and sum or average columns
    df = dummySample.groupby(['ID', 'ageBinTarget']).sum()
    temp = dummySample[['ID', 'sessionLength', 'viewingDifference']].groupby('ID').mean()
    del df['sessionLength']
    del df['viewingDifference']
    df['sessionLength'] = temp['sessionLength']
    df['viewingDifference'] = temp['viewingDifference']
    df = df.reset_index()

    #Add top 5 titles and encode
    df = df.merge(store, left_on='ID', right_index=True)
    df = pd.get_dummies(df, prefix=[5, 6, 7, 8, 9], columns=[5, 6, 7, 8, 9])


    # Split dataframe into features and target
    y = df.iloc[:, 1]  # .as_matrix()
    X = df.iloc[:, 2:]  # .as_matrix()

    # Scalings
    sc = StandardScaler()
    ma = MaxAbsScaler()
    mm = MinMaxScaler()


    # Apply scaler
    colNames = X.columns
    X.fillna(0, inplace=True)
    X = sc.fit_transform(X)
    X = pd.DataFrame(X, columns=colNames)

    # Remove features with less than 5% variance
    colNames = X.columns
    sel = VarianceThreshold(threshold=0.16)
    X = sel.fit_transform(X)
    # Get column names back
    newCols = []
    for remain, col in zip(sel.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)
    #X = transformed.merge(X.iloc[:, -5:], left_index=True, right_index=True)

    # Perform univariate feature selection (ANOVA F-values)
    colNames = X.columns
    selection_Percent = SelectPercentile(percentile=5)
    X = selection_Percent.fit_transform(X, y)
    # Get column names back
    newCols = []
    for remain, col in zip(selection_Percent.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)
    #X = transformed.merge(X.iloc[:, -5:], left_index=True, right_index=True)

    # Perform tree-based feature selection
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    colNames = X.columns
    sel = SelectFromModel(clf, prefit=True)
    X = sel.transform(X)
    newCols = []
    for remain, col in zip(sel.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    def testClassifier(clf):

        param_grid = [{'n_estimators': range(200, 500, 100),
                       #'max_depth': range(2, 8, 2),
                       'min_samples_split': range(2, 50, 15),
                       'min_samples_leaf': range(50, 400, 50),
                       'max_leaf_nodes': (20, 100, 20)
                       }]

        grid = GridSearchCV(clf, param_grid, cv=3, verbose=1, n_jobs=-1)
        fitted_classifier = grid.fit(X_train, y_train)
        print(grid.best_score_, grid.best_params_)
        predictions = fitted_classifier.predict(X_train)







        fitted = clf.fit(X_train, y_train)
        scoresCV = cross_val_score(clf, X_train, y_train, cv=3, verbose=0, n_jobs=-1)
        trainPredictionsCV = cross_val_predict(clf, X_train, y_train, cv=3, verbose=0, n_jobs=-1)

        trainPredictions = clf.predict(X_train)
        testPredictions = clf.predict(X_test)

        score1 = metrics.accuracy_score(y_test, testPredictions)
        score2 = metrics.cohen_kappa_score(y_test, testPredictions)
        #score3 = metrics.roc_auc_score(y_test, testPredictions)
        score4 = metrics.confusion_matrix(y_test, testPredictions)
        score5 = metrics.classification_report(y_test, testPredictions)
        print('Train score: ', metrics.accuracy_score(y_train, trainPredictions))
        print('CV score: ', scoresCV)
        print('Accuracy, Cohen Kappa')#, ROC AUC Score')
        print(score1, score2)#, score3)
        print('Confusion Matrix')
        print(score4)
        print('Classification Report')
        print(score5)

    lr = LogisticRegression()
    sgd = SGDClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(max_features='sqrt', max_depth=6)
    nb = GaussianNB()

    #print('LR')
    #testClassifier(lr)
    #print('DT')
    #testClassifier(dt)
    print('RF')
    testClassifier(rf)





    print('x')




# Split dataframe into features and target
    '''y = dfEncoded.iloc[:, 1]  # .as_matrix()
    X = dfEncoded.iloc[:, 2:]  # .as_matrix()

    # Scalings
    sc = StandardScaler()
    ma = MaxAbsScaler()
    mm = MinMaxScaler()

    # Apply scaler
    colNames = X.columns
    X = sc.fit_transform(X)
    X = pd.DataFrame(X, columns=colNames)

    # Remove features with less than 5% variance
    colNames = X.columns
    sel = VarianceThreshold(threshold=0.0475)
    X = sel.fit_transform(X)

    newCols = []
    for remain, col in zip(sel.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)

    # Perform univariate feature selection (ANOVA F-values)
    colNames = X.columns
    selection_Percent = SelectPercentile(percentile=5)
    X = selection_Percent.fit_transform(X, y)
    newCols = []
    for remain, col in zip(selection_Percent.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)

    # Perform tree-based feature selection
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    colNames = X.columns
    sel = SelectFromModel(clf, prefit=True)
    X = sel.transform(X)
    newCols = []
    for remain, col in zip(sel.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)'''


    print('x')




if __name__ == '__main__':
    main()