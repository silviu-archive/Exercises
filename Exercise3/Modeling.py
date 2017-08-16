import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, SelectFromModel, VarianceThreshold


def main():

    df = joblib.load('modelDataset.pkl')

    # Split dataframe into features and target
    y = df.iloc[:, 1]  # .as_matrix()
    X = df.iloc[:, 2:]  # .as_matrix()
    id = df.iloc[:, 0]

    # Scalings
    sc = StandardScaler()

    # Apply scaler
    colNames = X.columns
    X = sc.fit_transform(X)
    X = pd.DataFrame(X, columns=colNames)

    # Remove features with less than 20% variance
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
    clf = ExtraTreesRegressor()
    clf = clf.fit(X, y)
    colNames = X.columns
    sel = SelectFromModel(clf, prefit=True)
    X = sel.transform(X)
    newCols = []
    for remain, col in zip(sel.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    def testRegressor(clf):

        #grid = GridSearchCV(clf, param_grid, cv=3, verbose=1, n_jobs=-1)
        #fitted_classifier = grid.fit(X_train, y_train)
        #print(grid.best_score_, grid.best_params_)
        #predictions = fitted_classifier.predict(X_train)

        fitted = clf.fit(X_train, y_train)
        scoresCV = cross_val_score(clf, X_train, y_train, cv=3, verbose=0, n_jobs=-1)
        trainPredictionsCV = cross_val_predict(clf, X_train, y_train, cv=3, verbose=0, n_jobs=-1)

        trainPredictions = clf.predict(X_train)
        testPredictions = clf.predict(X_test)

        score1 = metrics.explained_variance_score(y_test, testPredictions)
        score2 = metrics.mean_absolute_error(y_test, testPredictions)
        score3 = metrics.mean_squared_error(y_test, testPredictions)
        score4 = metrics.r2_score(y_test, testPredictions)
        print('Train score: ', metrics.mean_absolute_error(y_train, trainPredictions))
        print('CV score: ', scoresCV)
        print('Explained Variance Score, MAE, MSE, R^2')
        print(score1, score2, score3, score4)

    lr = LinearRegression()
    rf = RandomForestRegressor()


    print('LR')
    testRegressor(lr)
    # print('DT')
    # testRegressor(dt)
    print('RF')
    testRegressor(rf)
    # print('et')
    # testRegressor(et)




if __name__ == '__main__':
    main()