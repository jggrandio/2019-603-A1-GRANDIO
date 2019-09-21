#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
using namespace std;

int* KNN(ArffData* dataset, int com)
{
	

	int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
	float* smallestDistance = (float*)malloc(com * sizeof(float));
	int* smallestDistanceClass = (int*)malloc(com * sizeof(int));

	float distance;
	float diff;

    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement KNN here, fill array of class predictions
		for(int i = 0; i < dataset->num_instances(); i++) // for each instance in the dataset
		{
			for (int l = 0; l<com; l++)
			{
				smallestDistance[l] = FLT_MAX; //first I initialise all the smallest to maxfloat
			}

				for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
    		{
					if(i == j) continue; //If It is the same instance I jump to the next iteration
					
					distance = 0; //Distance to 0 so the first one is a min distance
	
					for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
					{
						diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
						distance += diff * diff;
					}
					
					distance = sqrt(distance);
					for (int n = 0; n<com; n++)
					{
						if(distance < smallestDistance[n]) // select the closest one
						{
							for (int t=com-1; t>n; t--)
							{
								smallestDistance[t] = smallestDistance[t-1];
								smallestDistanceClass[t] = smallestDistanceClass[t-1];
							}
							smallestDistance[n] = distance;
							smallestDistanceClass[n] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
							break;
						}
					}
				}
				
				int freq = 0;
				int predict=0;
				for ( int m = 0; m<com; m++)
				{
					int tfreq = 1;
					int tpredict=smallestDistanceClass[m];
					for (int s = m+1 ; s<com; s++)
					{
						if (tpredict==smallestDistanceClass[s])
						{
							tfreq++;			
						}
					}
					if (tfreq>freq)
					{
						predict=smallestDistanceClass[m];
						freq=tfreq;
					}
				}


		predictions[i] = predict;
		}	
    	
		return predictions;
}

int* MPIKNN(ArffData* dataset, int com)
{
	int rank;
	int numtasks;
	
	
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	int space = (dataset->num_instances() / numtasks);
	int bias = (dataset->num_instances() % numtasks);
	int* predictions;
	int work_load;
	
	if(rank==numtasks-1)
	{
		work_load=space+bias;
		predictions = (int*)malloc((space+bias) * sizeof(int));
	}
	else
	{
		work_load=space;
		predictions = (int*)malloc(space * sizeof(int));
	}
	
	float* smallestDistance = (float*)malloc(com * sizeof(float));
	int* smallestDistanceClass = (int*)malloc(com * sizeof(int));

	float distance;
	float diff;

    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement KNN here, fill array of class predictions
		for(int i = space*rank; i < space*rank+work_load; i++) // for each instance in the dataset
		{
			for (int l = 0; l<com; l++)
			{
				smallestDistance[l] = FLT_MAX; //first I initialise all the smallest to maxfloat
			}

				for(int j = 0; j < dataset->num_instances(); j++) // target each other instance
    		{
					if(i == j) continue; //If It is the same instance I jump to the next iteration
					
					distance = 0; //Distance to 0 so the first one is a min distance
	
					for(int k = 0; k < dataset->num_attributes() - 1; k++) // compute the distance between the two instances
					{
						diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
						distance += diff * diff;
					}
					
					distance = sqrt(distance);
					for (int n = 0; n<com; n++)
					{
						if(distance < smallestDistance[n]) // select the closest one
						{
							for (int t=com-1; t>n; t--)
							{
								smallestDistance[t] = smallestDistance[t-1];
								smallestDistanceClass[t] = smallestDistanceClass[t-1];
							}
							smallestDistance[n] = distance;
							smallestDistanceClass[n] = dataset->get_instance(j)->get(dataset->num_attributes() - 1)->operator int32();
							break;
						}
					}
				}
				
				int freq = 0;
				int predict=0;
				for ( int m = 0; m<com; m++)
				{
					int tfreq = 1;
					int tpredict=smallestDistanceClass[m];
					for (int s = m+1 ; s<com; s++)
					{
						if (tpredict==smallestDistanceClass[s])
						{
							tfreq++;			
						}
					}
					if (tfreq>freq)
					{
						predict=smallestDistanceClass[m];
						freq=tfreq;
					}
				}


		predictions[i-rank*space] = predict;
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
  
		return predictions;
}
int* MPIcomputeConfusionMatrix(int* predictions, ArffData* dataset)
{
	int rank;
	int numtasks;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	int space = (dataset->num_instances() / numtasks);
	int bias = (dataset->num_instances() % numtasks);
	int work_load;
	
	if(rank==numtasks-1)
	{
		work_load=space+bias;
	}
	else
	{
		work_load=space;
	}
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
    
    for(int i = rank*space; i < rank*space+work_load; i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i-rank*space];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return confusionMatrix;
}

float MPIcomputeAccuracy(int* confusionMatrix, ArffData* dataset)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
	int totalSucces;
    MPI_Reduce(&successfulPredictions, &totalSucces, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	MPI_Finalize();
	if (rank == 0)
	{
		return totalSucces / (float) dataset->num_instances();
	}
    else
	{
		exit(0);
	}
}




	
int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
		cout << "Usage: k value" << endl;
        exit(0);
    }
    
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
	
	int k;
	
	sscanf(argv[2], "%d", &k);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    int* predictions = MPIKNN(dataset,k);
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int* confusionMatrix = MPIcomputeConfusionMatrix(predictions, dataset);
    float accuracy = MPIcomputeAccuracy(confusionMatrix, dataset);
    
    
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
  
    printf("The MPIKNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    predictions = KNN(dataset,k);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    confusionMatrix = computeConfusionMatrix(predictions, dataset);
    accuracy = computeAccuracy(confusionMatrix, dataset);

    printf("The KNN classifier  for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
