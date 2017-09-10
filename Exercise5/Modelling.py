import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn import metrics

from DataProcessing import preprocessData

def trainModel():

    df = preprocessData()

    #Split train and test set
    dfTest = df.loc[df['TestSet'] == 1]
    df = df.loc[df['TestSet'] == 0]

    # Split dataframe into target and features
    y = df.iloc[:, 0]  # .as_matrix()
    X = df.iloc[:, 1:]  # .as_matrix()

    # Apply standard scaler in order to remove mean and scale to unit variance (so large-valued features won't
        #heavily influence the model)
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

    # Perform univariate feature selection (ANOVA F-values)
    colNames = X.columns
    selection_Percent = SelectPercentile(percentile=20)
    X = selection_Percent.fit_transform(X, y)
    # Get column names back
    newCols = []
    for remain, col in zip(selection_Percent.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)

    # Perform tree-based feature selection
    clf = ExtraTreeClassifier()
    clf = clf.fit(X, y)
    colNames = X.columns
    sel = SelectFromModel(clf, prefit=True)
    X = sel.transform(X)
    newCols = []
    for remain, col in zip(sel.get_support(), colNames):
        if remain == True:
            newCols.append(col)
    X = pd.DataFrame(X, columns=newCols)


    # Tranform test set to contain same features as train set
    dfTestTarget = dfTest['#40 (target) nominal']
    dfTest = dfTest[X.columns]

    #Define train/test variables
    X_train = X
    y_train = y
    X_test = dfTest
    y_test = dfTestTarget

    def testClassifier(clf):

        '''
        #XGB tuning - concept, not in use
        param_grid = [{'max_depth': range(2, 4, 1),
                       'min_child_weight': range(3, 6, 1),
                       'n_estimators': range(80, 110, 10),
                       'learning_rate': [0.1],
                       'gamma': [0],
                       'subsample': [0.9, 1],
                       'colsample_bytree': [0.7],
                       'reg_alpha': [15, 50, 100, 150, 200],
                       'reg_lambda': [15, 20, 25, 30, 40, 50]}]
        fit_params = {"early_stopping_rounds": 8,
                      "eval_metric": "mae",
                      "eval_set": [[X_test, y_test]],
                      "verbose": False}
        grid = GridSearchCV(clf, param_grid, fit_params=fit_params,
                            cv=3, verbose=1, n_jobs=-1)
        fitted_classifier = grid.fit(X_train, y_train)
        print(grid.best_score_, grid.best_params_)
        predictions = fitted_classifier.predict(X_train)
        '''

        fitted = clf.fit(X_train, y_train)
        scoresCV = cross_val_score(clf, X_train, y_train, cv=3, verbose=0, n_jobs=-1)
        trainPredictionsCV = cross_val_predict(clf, X_train, y_train, cv=3, verbose=0, n_jobs=-1)

        trainPredictions = clf.predict(X_train)
        testPredictions = clf.predict(X_test)


        score1 = metrics.accuracy_score(y_test.values, testPredictions)
        #score2 = metrics.auc(y_test.values, testPredictions)
        score3 = metrics.cohen_kappa_score(y_test.values, testPredictions)
        score4 = metrics.classification_report(y_test.values, testPredictions)
        print('Train score: ', metrics.accuracy_score(y_train.values, trainPredictions))
        print('CV score: ', scoresCV)
        print('Accuracy score, AUC, Cohen Kappa, Classification Report')
        print(score1, score3, score4)

        #tempIndex = range(0, len(y_test.values), 1)
        #plt.scatter(tempIndex, y_test.values, color='black', s = 20, alpha=0.8)
        #plt.scatter(tempIndex, testPredictions, color='red', s = 20, alpha=0.4)
        #plt.show()

        #Results appear to be highly interesting
        #MSE (and thus penalising large errors more) suggests that the model does not deal well with
            #particular categories of retweets where there is a significant difference between true value and predicted
        #Data appears to have high bias in terms of selection, as if tweets were selected from specific pools
            #based on retweet value
        #While the random forest deals well with those particular types of tweets, more analysis is needed
        # Further steps would start by understanding the sampling procedure that produced these tweets
            # From there, features need to be relooked at, dimensionality reduction (such as PCA) might be needed
            # Simpler / more powerful models to then be appropriately applied
        #The target retweets actually seem to be created from a Decision Tree Model
        print('x')




    print('DT')
    dt = DecisionTreeClassifier()
    testClassifier(dt)
    #print('RF')
    #testClassifier(dt)
    #print('XGB')
    #testClassifier(gb)




if __name__ == '__main__':
    trainModel()