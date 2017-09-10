import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier, export_graphviz
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from imblearn.combine import SMOTEENN

import xgboost

from DataProcessing import preprocessData

def trainModel():

    #Read processed dataframe
    df = preprocessData()

    #Move test set indicator to the 2nd position of the dataframe
    cols = list(df)
    cols.insert(1, cols.pop(cols.index('TestSet')))
    df = df.ix[:, cols]

    # Split dataframe into target and features
    y = df.iloc[:, 0]  # .as_matrix()
    flag = pd.DataFrame(df.iloc[:, 1]) # .as_matrix()
    X = df.iloc[:, 2:]  # .as_matrix()

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

    #Perform dimensionality reduction using PCA
    pca = PCA(n_components=5)
    pca.fit(X)
    #PCA scree plot - aid in determining number of components
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    #plt.show()

    #Create PCA dataframe and append to original
    #Adding principle components adds additional insight to the dataframe
    #If PCs do not perform well, they will be removed in further feature selection procedures
    dfPCA = pd.DataFrame(pca.transform(X))
    newCols = []
    for col in dfPCA.columns:
        name = 'PCA' + str(col)
        newCols.append(name)
    dfPCA.columns = newCols
    X = pd.merge(X, dfPCA, left_index=True, right_index=True)

    # Perform univariate feature selection (ANOVA F-values)
    colNames = X.columns
    selection_Percent = SelectPercentile(percentile=50)
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

    #Split train and test set
    #Create new test set column in X
    X['TestSet'] = flag['TestSet'].tolist()
    X['Target'] = y.tolist()
    #Encode target (to binary) - for ROC AUC metric (0 for under 50k, 1 for over)
    le = LabelEncoder()
    X['Target'] = le.fit_transform(X['Target'])
    #Copy in dfTest all the test set values from X
    dfTest = X.loc[X['TestSet'] == 1]
    #Re-write X with only learning set values
    X = X.loc[X['TestSet'] == 0]
    #Define test set target
    dfTestTarget = dfTest['Target']
    #Remove target and 'test set' column from test dataframe
    dfTest.drop(['TestSet', 'Target'], axis=1, inplace=True)
    #Create new learning target series
    y = X['Target']
    #Drop newly inserted columns from learning dataframe
    X.drop(['TestSet', 'Target'], axis=1, inplace=True)
    #Retain column names
    colNames = X.columns

    # The dataset is heavily imbalanced in terms of classes, and balancing procedures need to be conducted
    # Testing various under / over / combined sampling procedures
    # Some of these procedures are very computationally expensive (and thus are not suitable for home use e.g. SMOTEENN)
    rus = RandomUnderSampler()
    X, y = rus.fit_sample(X, y)
    #sme = SMOTEENN(n_jobs=-1)
    #X, y, = sme.fit_sample(X, y)
    X = pd.DataFrame(X, columns=colNames)
    y = pd.Series(y, name='Target')

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
        #X_test['Predictions'] = testPredictions

        score1 = metrics.accuracy_score(y_test.values, testPredictions)
        score2 = metrics.roc_auc_score(y_test.values, testPredictions)
        score3 = metrics.cohen_kappa_score(y_test.values, testPredictions)
        score4 = metrics.classification_report(y_test.values, testPredictions)
        print('Train score: ', metrics.accuracy_score(y_train.values, trainPredictions))
        print('CV score: ', scoresCV)
        print('Accuracy score, ROC AUC, Cohen Kappa')
        print(score1, score2, score3)
        print('Classification Report')
        print(score4)

        #WITH UNDER-SAMPLING
        #Low Precision in Class 1 (~0.28) = suggests that too many salaries are labeled as >50k when they are <50k
            #Could be a potential after-effect of under-sampling
        #High Recall in Class 1 (~0.90) = suggests that the classifier is able to find all positive samples

        #WITHOUT UNDER-SAMPLING
        #High Precision in Class 1 (~0.76) = suggests that the classifiers handles negative samples well
        #Low Recall in Class 1 (~0.39) = suggests that the classifier is not able to find all positive samples

        return clf

    print('LR')
    lr = LogisticRegression(C = 100)
    clf = testClassifier(lr)
    print('DT')
    dt = DecisionTreeClassifier()
    clf = testClassifier(dt)
    export_graphviz(clf, out_file = 'tree.dot')
    print('RF')
    rf = RandomForestClassifier()
    clf = testClassifier(rf)
    print('XGB')
    gb = xgboost.XGBClassifier()
    clf = testClassifier(gb)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    trainModel()