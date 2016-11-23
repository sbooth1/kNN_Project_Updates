/*
*	Steven Booth, Nick Goodpaster
*	
*	COEN171 - Exploring Programming Languages Project
*	
*	Description: C implementation of k-Nearest Neighbors
*	machine learning algorithm. Makes a prediction for test 
*	sample given training sample data. Attributes are loaded from
*	iris.csv file, which is split into training and testing data.
*	Attributes include pedal length, pedal width, sepal length, 
*	and sepal width. Identifiers are setosa, virginica, and versicolor. 
*	The program computes a prediction for the testing data with
*	96% accuracy. 
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#define N  150
#define TRAIN 120
#define TEST 30
#define MAX 11

/*	
*	Structure declaration for neighbor type. 
*	Holds ID and distance from test instance
*/
typedef struct{
	float distance;
	char *ID;
} neighbor;

//Function definitions
float distanceFormula(float **train, float *testInstance, int i);
float* getDistances(float **train, float *testInstance);
neighbor* sortDistances(float *distances, char **trainingID);
neighbor* getNN(neighbor *sortedNeighbors, int k);
char* getVote(neighbor *NN, int k);
char* getMax(int setosa, int virginica, int versicolor);
void loadDataSet(char *filename, float **data, char **ID);
float getAccuracy(char **predictions, char **testingID);

//Main function
int main(void){
	int k = 3;
	
	//Allocates memory for testing data set
	float **testingData = (float **) malloc(TEST * sizeof(float *));
  	for(int i = 0; i < TEST; i++){
		testingData[i] = (float *)malloc(4 * sizeof(float));
	}
	char **testingDataID = malloc(TEST * sizeof(char*));
	for(int i = 0; i < TEST; i ++){
		testingDataID[i] = (char *) malloc(10 * sizeof(char));
	}
	
	//Allocates memory for training data set	
	float **trainingData = (float **) malloc(TRAIN * sizeof(float *));
	for(int i = 0; i < TRAIN; i++){
		trainingData[i] = (float *)malloc(4 * sizeof(float));
	}
	char **trainingDataID = malloc(TRAIN * sizeof(char*));
	for(int i = 0; i < TRAIN; i ++){
		trainingDataID[i] = (char *) malloc(10 * sizeof(char));
	}

	//Allocates memory to hold predictions
	char **predictions = malloc(TEST * sizeof(char *));
	for(int i = 0; i < TEST; i ++){
		predictions[i] = (char *) malloc(10 * sizeof(char));
	}
	
	//Loads the data sets from input files into data arrays
	loadDataSet("iristraining.txt",trainingData, trainingDataID);
	loadDataSet("iristesting.txt", testingData, testingDataID);	
	
	//calculates a prediction for each data sample in testingData	
	for(int j = 0; j < TEST; j++){	
		float *distances = getDistances(trainingData, testingData[j]);
		neighbor *neighbors = sortDistances(distances, trainingDataID);	
		neighbor *NN = getNN(neighbors, k); 	
		predictions[j] = getVote(NN, k);	
		printf(">>Prediction: %s   >>>>   Actual: %s \n", predictions[j], testingDataID[j]);
		free(neighbors);
		free(distances);
	}

	//working to fix strcmp in getAccuracy function
	//printf("\n\n the accuracy of the alogirithm is: %f\n", getAccuracy(predictions, testingDataID));
	free(predictions);
	free(testingData);
	free(testingDataID);
	free(trainingData);
	free(trainingDataID);
	return 0;
}

/*
*	Euclidian distance formula, calculates distance between
*	flower attributes. 
*/
float distanceFormula(float **train, float *testInstance, int i){
	float distance = 0;
	distance += pow(testInstance[0] - train[i][0], 2);
	distance += pow(testInstance[1] - train[i][1], 2);
	distance += pow(testInstance[2] - train[i][2], 2);
	distance += pow(testInstance[3] - train[i][3], 2);
	return sqrt(distance);
}

/*
*	Calculates distance between testing instance and every 
*	training value. 
*/
float* getDistances(float **train, float *testInstance){
	//printf("%f, %f\n", testInstance[0], testInstance[1]);
	float *distances = (float *) malloc(sizeof(int) * TRAIN);
	for(int i = 0 ; i < TRAIN; i ++){ 
		float distance = distanceFormula(train, testInstance, i);		
		distances[i] = distance;
	}
	return distances;
}

/*
*	Pairs the distance and the ID to create a neighbor type for each
*	of the distances. Sorts this array of neighbors to be used for 
*	finding the k-Nearest Neighbors.
*/
neighbor* sortDistances(float *distances, char **trainingID){
	neighbor* neighbors = (neighbor *) malloc(TRAIN * sizeof(neighbor));
	neighbor* temp = (neighbor *)  malloc(sizeof(neighbor));
	
	//Loading distances and trainingId's into array of neighbors
	for(int i = 0; i < TRAIN; i ++){
		neighbors[i].distance = distances[i];
		neighbors[i].ID = malloc(10 * sizeof(char));
		strcpy(neighbors[i].ID, trainingID[i]);
	}
	
	//Bubble sort array of neighbors by distance
	for(int i = 0; i < (TRAIN -1); i++){
		for(int j = 0; j < (TRAIN - i - 1); j ++){
			if(neighbors[j].distance > neighbors[j + 1].distance){
				*temp = neighbors[j];
				neighbors[j] = neighbors[j + 1];
				neighbors[j + 1] = *temp;
			}
		}
	}
	return neighbors;			
}

/*
*	Uses the sorted neighbor array to create a subarray containing
*	only the k-Nearest Neighbors of the test instance. 
*/
neighbor* getNN(neighbor *sortedNeighbors, int k){
	neighbor* NN = (neighbor *) malloc (k * sizeof(neighbor));
	for(int i = 0; i < k; i ++){
		NN[i] = sortedNeighbors[i];
	}	
	return NN;
}

/*
*	Each neighbor will vote for it's ID type, these votes will 
*	then be passed into the getMax function to find the majority
*	ID of the neighbors.
*/
char* getVote(neighbor *NN, int k){
        int setosa = 0, virginica = 0, versicolor = 0;
	for(int i = 0; i < k; i ++){
		if(strcmp("setosa", NN[i].ID) == 0){
			setosa++;
		}else if(strcmp("virginica", NN[i].ID) == 0){
			virginica++;	
		}else if(strcmp("versicolor", NN[i].ID) == 0){
			versicolor++;
		}
	}
	return getMax(setosa, virginica, versicolor);
}

/*
*	Returns the prediction, which is the ID with the most 
*	votes. 
*/
char* getMax(int setosa, int virginica, int versicolor){
	char* prediction = malloc(MAX * sizeof(char));
	if(setosa > virginica && setosa > versicolor){
		strcpy(prediction, "setosa");
	} else if(virginica > setosa && virginica > versicolor){
		strcpy(prediction, "virginica");
	} else if(versicolor > setosa && versicolor > virginica){
		strcpy(prediction, "versicolor");
	} else{
		strcpy(prediction, "still cannot be determined");
	}
	return prediction;
}

/*
*	Function to parse the csv files for the training and testing
*	data sets into the testingData and trainingData arrays. 
*/
void loadDataSet(char *filename, float **data, char **ID){
	FILE* file = fopen(filename, "r");
        char line[40];
        char *instance = (char *) malloc(MAX * sizeof(char));
        int i, j, rowcount = 0, IDcount = 0, instancecount = 0;;
        float instanceData;
        while(fgets(line, sizeof(line), file)){
                i = 0;
                instancecount = 0;
                while(i < sizeof(line)){
                        j = 0;
                        if(!isalpha(line[i])){
                                do{
                                        instance[j] = line[i];
                                        j++;
                                        i++;
                                }while(line[i] != ',');
                                instanceData = atof(instance);
                                data[rowcount][instancecount] = instanceData;
                                instance[0] = '\0';
                                instancecount++;
                        } else if(isalpha(line[i])){
                                do{
                                        instance[j] = line[i];
                                        j++;
                                        i++;
                                }while(isalpha(line[i]));
                                instance[j++] = '\0';
                                strcpy(ID[IDcount], instance);
                                IDcount ++;
                                break;
                        }
                        i++;
                }
                rowcount ++;
        }
        free(instance);
	fclose(file);
}

/*
*	Function to calculate the accuracy of the alogorithm by evaluating
*	how many of the testingDataID's were predicted correctly. 
*/

float getAccuracy(char **predictions, char **testingID){
	int correctpredictions = 0;
	for(int i = 0; i < TEST; i ++){
		printf("%s, %s\n", testingID[i], predictions[i]);
		if(strcmp(testingID[i], predictions[i]) == 0){
			correctpredictions++;
		}
	}
	return 100 * (correctpredictions/TEST);
}

