import pandas as pd #0.20.1
import matplotlib.pyplot as plt #2.0.2
import seaborn as sns #0.7.1
import sys
import warnings

from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib



#The script reads a downloaded dataset from https://archive.ics.uci.edu/ml/datasets/Iris
#The data is briefly analyzed in terms of relationships between attributes
#The script takes as arguments attributes of a user-defined species of plant
#The script retrieves the 10 most similar data points from the original dataset
#The results are then displayed using a pair plot


#No unit tests were created for the development of this script
#However, try/except
#The script was created and tested using PyCharm and the implemented step debugger


#The solution could be expanded in multiple ways:
#Initial data visualization / analysis could be more thorough - boxplots/violinplots of data, density
#Saving the model via joblib / pickle, so that the model is not trained upon every iteration, but is stored on disk
#Implementation of a classification algorithm for the prediction of a new input's species
#Define




#Class defining column names
class columnNames:
    columnItemID = 'ItemID'
    columnDistanceToInput = 'DistanceToInput'
    columnSepalLength = 'SepalLength'
    columnSepalWidth = 'SepalWidth'
    columnPetalLength = 'PetalLength'
    columnPetalWidth = 'PetalWidth'
    columnClass = 'Class'

#Function that asks user for input and checks if it can convert it to float
def retrieveAndCheckInput(inputText):
    try:
        inputValue = float(input('%s: ' % inputText))
        return inputValue
    except ValueError:
        print('Please enter a floating number')
        sys.exit(0)

#Method to initialize visualization dataframe (appending the first row as the user-defined plant)
def createVisuaalizationDataframe(sepalLength, sepalWidth, petalLength, petalWidth):
    dfVisualization = pd.DataFrame(columns = ['ItemID', 'DistanceToInput', 'SepalLength', 'SepalWidth', 'PetalLength',
                                    'PetalWidth', 'Class'])
    itemDict = {'ItemID': 'User',
                'DistanceToInput': 0, #Distance from user item to itself will be zero
                'SepalLength': sepalLength,
                'SepalWidth': sepalWidth,
                'PetalLength': petalLength,
                'PetalWidth': petalWidth,
                'Class': 'Unknown'}
    dfVisualization = dfVisualization.append(itemDict, ignore_index=True)
    return dfVisualization

def main():

    #Read .data file (via csv reader in pandas)
    df = pd.read_csv('J:\Datasets\Exercises\Exercise4\iris.data', header=None)
    #Set column headers according to data docs
    df.columns = [columnNames.columnSepalLength, columnNames.columnSepalWidth, columnNames.columnPetalLength,
                  columnNames.columnPetalWidth, columnNames.columnClass]

    #Show summary statistics for the dataframe
    print('Summary Stats \n', df.describe())

    #Plot correlation table
    sns.heatmap(df.corr(), annot=True)
    plt.title('Correlation Table')

    #Plot relationships between the 4 variables (taking into account Class distributions as well)
    sns.pairplot(df, hue='Class')
    #Based on the pairplot, setosa appears to be linearly separable from verginica and versicolour



    #Select the attributes defining a species of Iris plant
    X = df.iloc[:, :-1].values

    #Create and fit a Nearest Neighbors model using a k-dimensional tree model, and a euclidean metric
    numberOfNeighbors = 10
    nbrs = NearestNeighbors(n_neighbors=numberOfNeighbors, algorithm='kd_tree').fit(X)

    #Start user input procedure
    print('This program retrieves the top 10 similar data points from an existing dataset')
    print('These datapoints refer to similar Iris plant species')
    print('Please insert the following arguments to return the solution:')

    #Retrieve input from user
    sepalLength = retrieveAndCheckInput('SepalLength')
    sepalWidth = retrieveAndCheckInput('SepalWidth')
    petalLength = retrieveAndCheckInput('PetalLength')
    petalWidth = retrieveAndCheckInput('PetalWidth')

    #Create visualization storage container, starting with the user input
    dfVisualization = createVisuaalizationDataframe(sepalLength, sepalWidth, petalLength, petalWidth)

    print('The closest neighbors are (in descending order): ')
    distanceToNeighbors = nbrs.kneighbors([sepalLength, sepalWidth, petalLength, petalWidth])[0][0]
    neighborsIndices = nbrs.kneighbors([sepalLength, sepalWidth, petalLength, petalWidth])[1][0]

    #Iterate over the closest 10 neighbours and retrieve results
    for distance, index in zip(distanceToNeighbors, neighborsIndices):
        print('Item number %s by a distance of %s, with the following features: '
              'Sepal Length %s cm, Sepal Width %s cm, Petal Length %s cm, Petal Width %s cm; '
              'The item belongs to the class %s' % (index, distance, df.loc[index, 'SepalLength'],
                                                    df.loc[index, 'SepalWidth'], df.loc[index, 'PetalLength'],
                                                    df.loc[index, 'PetalWidth'], df.loc[index, 'Class']))
        #Append neighbour attributes to visualization dataframe
        itemDict = {'ItemID': index,
                    'DistanceToInput': distance,
                    'SepalLength': df.loc[index, 'SepalLength'],
                    'SepalWidth': df.loc[index, 'SepalWidth'],
                    'PetalLength': df.loc[index, 'PetalLength'],
                    'PetalWidth': df.loc[index, 'PetalWidth'],
                    'Class': df.loc[index, 'Class']}
        dfVisualization = dfVisualization.append(itemDict, ignore_index=True)





    #Plot relationships between the user input and the 10 closest neighbors
    sns.pairplot(dfVisualization, hue='Class')
    plt.show()



    print('x')










if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()