
"""
Steven Booth, Nick Goodpastor

COEN 171 - Exploring Programming Languages Project

Algorithm: k-Nearest Neighbors

Description: This alogirthm will take in the iris.csv file and parse the data 
into training and testing data. Once this is complete, each instance of testing
data will be compared to every training data instance and the distance between
the attributes will be calculated and stored in a distance array. Next, the 
k-nearest neighbors of the test instance will take a vote using their own 
identifiers. The most common identifier amongst the nearest neighbors will be
the prediciton for the test instance. Once all the predictions are made, the 
program will compute its accuracy based on the actual identifers of the testing
data set and the predicted identifiers. Our program runs with a 95% or better
accuracy when using 5 or less neighbors. 

"""

#imported libraries
import math
import csv
import random
import operator

"""
Function to load the data set from the irisdata file and split the data into
training and testing data records. 
"""

def loadDataSet(filename, split, trainingDataSet = [], testingDataSet = []):
    with open(filename, 'r') as csvfile:
        #Reads lines of the csv file using the built in csv reader
        lines = csv.reader(csvfile)
        #makes a list of the lines in the csv file
        data_csv = list(lines)
        for x in range(len(data_csv)):
            #for each of the attributes in the csv file, the values are cast
            #to floats
            for y in range(5):
                data_csv[x][y] = float(data_csv[x][y])
            #creates a random number between 0 and 1.0 and appends value
            #to training or testing data set with a weighted split
            if random.random() < split:
                trainingDataSet.append(data_csv[x])
            else:
                testingDataSet.append(data_csv[x])

"""
Function to compute the distance between a testingInstance and trainingData.

Returns the distance between the testing attribute and the training attribute
"""

def distanceFormula(testingInstance, trainingData, length):
    distance = 0
    for x in range(length):
        distance += pow((float(testingInstance[x])) - (float(trainingData[x])), 2)
    return math.sqrt(distance)
    
    
"""
Function to compute the k-Nearest Neighbors. Takes in a test instance and 
compares the distance between its attributes and the attributes of all the
instance values in the training data. Takes the closest k neighors and returns 
an array containing the k-Nearest Neighbors neighbors. 

Returns the array of nearest neighbors. 
"""
def getNN(trainingDataSet, testInstance, k):
    print('\n')
    #creates an array to hold distances for neighbors
    distances = []
    #sets the length to be the number of attributes, so the number of columns
    #in the test instance minus the identifier for the type of flower
    length = len(testInstance) - 1
    for x in range(len(trainingDataSet)):
        #computes distance between training instance and testing instance
        distance = distanceFormula(testInstance, trainingDataSet[x], length)
        #appends to distances array
        distances.append((trainingDataSet[x], distance))
    #sorts the distances
    distances.sort(key=operator.itemgetter(1))
    neighbors= []
    #appends the k-Nearest neighbors to the neighbors array 
    for x in range(k):
        neighbors.append(distances[x][0])
        print('Neighbor #' + repr(x + 1) + ': ' + repr(neighbors[x][-1]))
    return neighbors
    
"""
Function to get the votes from the nearest neighbors to decide the predicted
type of the testing instance. Will use the getMax function to generate a 
prediction.

Returns string representation of prediction to be added to the predictions 
array in the main. 
"""
def getVotes(neighbors):
    setosaVotes = 0
    virginicaVotes = 0
    versicolorVotes = 0
    for x in range(len(neighbors)):
        if neighbors[x][5] == 'setosa':
            setosaVotes += 1
        if neighbors[x][5] == 'virginica':
            virginicaVotes += 1
        if neighbors[x][5] == 'versicolor':
            versicolorVotes += 1
    prediction = getMax(setosaVotes, virginicaVotes, versicolorVotes)
    return prediction
  
"""
Function to get the maximum value of the three votes and returns the string
representation of the most voted flower type from the neighbors array. 

Returns the prediction in the form of a string. 
"""  

def getMax(setosaVotes, virginicaVotes, versicolorVotes):
    if((setosaVotes > virginicaVotes) & (setosaVotes > versicolorVotes)):
        prediction = 'setosa'
    if((virginicaVotes > setosaVotes) & (virginicaVotes > versicolorVotes)):
        prediction = 'virginica'
    if((versicolorVotes > setosaVotes) & (versicolorVotes > virginicaVotes)):
        prediction = 'versicolor'
    return prediction
          
"""
Function to compute the accuracy of the algorithm by comparing the correct
flower types from the testing data and the predictions computed in our
algorithm.

Returns the value of the accuracy as a float to be printed int the main
function. 
"""
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][5] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100
    
"""
Main function definition.
"""
def main():
    #prepare data
    trainingDataSet = []
    testingDataSet = []
    split = 0.8
    loadDataSet('iris.csv', split, trainingDataSet, testingDataSet)
    #generate predictions
    predictions=[]
    k = 3
    for x in range(len(testingDataSet)):
        neighbors = getNN(trainingDataSet, testingDataSet[x], k)
        result = getVotes(neighbors)
        predictions.append(result)
        print('Predicted=' + repr(result) + ', Actual=' + repr(testingDataSet[x][-1]))
        print('-------------------------------------------')
    accuracy = getAccuracy(testingDataSet, predictions)
    print('\n\nAlgorithm Accuracy: ' + repr(accuracy) + '%')
    
#main function call     
main()

