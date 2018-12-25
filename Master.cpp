#pragma warning (disable: 4996)
#include "Master.h"
#include "Structs_MPI.h"
#include <omp.h>

void MasterProccess(int numproc, Params *params, Point *points, char *outputFileName)
{
	// init allData
	AllData allData;
	allData.parameters = params;
	allData.points = points;
	allData.isMasterFoundResult = FALSE;
	allData.isSlavesListenerFoundResult = FALSE;

	// every Result type in allData have k clusters, and each cluster has memory for N points' indices (the maximal value)
	initMemoryResultsForAllData(&allData);
	
	Search_Result(numproc, &allData);




	
	// End of the search. All slaves are free, and now the master need to understand which result is the best one (or the only one)

	if (!allData.isMasterFoundResult && !allData.isSlavesListenerFoundResult)
	{
		// if there is no result at all:
		printf("\nThere is no solution!");
		writeNoResultToFile(outputFileName);
		fflush(stdout);
	}
	else
		if (allData.isMasterFoundResult && allData.isSlavesListenerFoundResult) // master and slaves' listener found results
		{
			// if master result is better
			if (allData.masterResult.PICTURE_ID < allData.slavesListenerResult.PICTURE_ID)
			{
				printPicture(&allData.masterResult, allData.parameters->K);
				writeSuccessResultToFile(outputFileName, &allData.masterResult, params);
			}
				
			else
			{
				// slaves' listener' result is better
				printPicture(&allData.slavesListenerResult, allData.parameters->K);
				writeSuccessResultToFile(outputFileName, &allData.slavesListenerResult, params);
			}
		}
	else  // one of them got a good result, and the other didn't
	{
		if (!allData.isMasterFoundResult)
		{
			printPicture(&allData.slavesListenerResult, allData.parameters->K);
			writeSuccessResultToFile(outputFileName, &allData.slavesListenerResult, params);
		}
		else
		{
			printPicture(&allData.masterResult, allData.parameters->K);
			writeSuccessResultToFile(outputFileName, &allData.masterResult, params);
		}
			
	}
		
	// free all clusters' points indices in each result: master and slaves' listener
	int i;
	Cluster *currentMasterCluster;
	Cluster *currentSlavesListenerCluster;
	for (i = 0; i < params->K; i++)
	{
		currentMasterCluster = &(allData.masterResult.clusters[i]);
		currentSlavesListenerCluster = &(allData.slavesListenerResult.clusters[i]);

		free(currentMasterCluster->pointsIndices);
		free(currentSlavesListenerCluster->pointsIndices);
	}
	free(allData.masterResult.clusters);
	free(allData.slavesListenerResult.clusters);
}  // end Master turn

/*
	create 2 results for allData: one for master calc and one for slaves' listener
*/
void initMemoryResultsForAllData(AllData *allData)
{
	Params *params = allData->parameters;
	Result* masterResult = &allData->masterResult;
	Result* slavesListenerResult = &allData->slavesListenerResult;
	
	masterResult->clusters = (Cluster*)calloc(params->K, sizeof(Cluster));
	slavesListenerResult->clusters = (Cluster*)calloc(params->K, sizeof(Cluster));

	int i;
	Cluster *currentMasterCluster;
	Cluster *currentSlavesListenerCluster;
	for (i = 0; i < params->K; i++)
	{
		currentMasterCluster = &masterResult->clusters[i];
		currentSlavesListenerCluster = &slavesListenerResult->clusters[i];

		currentMasterCluster->pointsIndices = (int*)calloc(params->N, sizeof(int));
		currentSlavesListenerCluster->pointsIndices = (int*)calloc(params->N, sizeof(int));
	}
}


/*
Purpose:
Divide the work between two main threads by sections (Parallism).
The first section send to slaves information and recv from slaves information. Moreover, he communicate with the other thread.
The second one calc a picture, like any other slave.
Important remark:
Those both threads belong to Master.

The first section provides the ability to MASTER to listen to and control on the slaves and their results
and the second one provides the ability to MASTER to calc the picture by himself.
*/
void Search_Result(int numproc, AllData *allData)
{
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			//master' own dt search
			Master_Search(allData);
		}
		
		#pragma omp section
		{
			//talk to other machine slaves
			activateSlavesListener(numproc, allData);
		}
	}
}


/*
Proccess:
- getting the next picture' ID
- while an optimal result was not found, AND the picture' ID is legal, do:
- calc the picture and find result, which means: find Q and clusters
- if Q is good -> set the optimal result in allData (By method)
else -> get the next picture id
Purpose:
- Finding a good result owing to the Master' resources.
*/
void Master_Search(AllData *allData)
{
	Result* masterResult = &allData->masterResult;
	int numOfPictures;
	int isContinue = TRUE;
	numOfPictures = getNumOfPictures(allData->parameters);

	// getting next picture to calc:
	masterResult->PICTURE_ID = getNextPictureID();

	while (isContinue)
	{
		/*
			master loop until good q achivied or out of time
			Calc_pircute_parallel : calculate q and find diameter
			check if q is good enough based on QM read from file
		*/
		printf("\nMaster start calc picture=%d\n", masterResult->PICTURE_ID);
		fflush(stdout);
		Calc_Picture_Parallel(allData->parameters, allData->points, masterResult);

		if (isQok(masterResult, allData))  // if the picture is good:
		{
			allData->isMasterFoundResult = TRUE;
			isContinue = FALSE;  // stop the 
		}	
		else  // else: the picture isn't good:
		{
			if(allData->isSlavesListenerFoundResult)  // if slaves' listener found a solution
				isContinue = FALSE;
			else  // else: there isn't a good result at all
			{
				masterResult->PICTURE_ID = getNextPictureID(); 
				if (masterResult->PICTURE_ID >= numOfPictures)  // if there are no pictures to calc
					isContinue = FALSE;
			}
		}
	}
}

int isQok(Result *myResult, AllData* allData)
{
	double Q = myResult->Q;
	double QM = allData->parameters->QM;
	if (Q != NO_SOLUTION_YET && Q < QM)
	{
		return TRUE;
	}
	return FALSE;
}

static void printPicture(Result *result, int numOfClusters)
{
	printf("\nPicture ID is: %d", result->PICTURE_ID);
	printf("\nQ is: %lf", result->Q);
	printf("\nClusters Centers: \n");
	printClusters(result->clusters, numOfClusters);
}

static void printClusters(Cluster *clusters, int numOfClusters)
{
	int i;
	Vector* currentClusterLocation;
	for (i = 0; i < numOfClusters; i++)
	{
		currentClusterLocation = &(clusters[i].center);
		printVector(currentClusterLocation);
	}
}

static void printVector(Vector *vector)
{
	printf("(%lf, %lf, %lf)\n", vector->x, vector->y, vector->z);
}



////////////////////****************WRITE****************////////////////////////
void writeSuccessResultToFile(char *fileName, Result *result, Params *params)
{
	FILE* f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("\nwriting to file was failed");
		return;
	}

	fprintf(f, "\nQ = %lf", result->Q);

	double time = params->dT * result->PICTURE_ID;
	fprintf(f, "\nt = %lf", time);

	fprintf(f, "\nCenters of the clusters:");
	int i;
	for (i = 0; i < params->K; i++)
		printClusterToFile(f, &result->clusters[i]);
}

void printClusterToFile(FILE* f, Cluster *cluster)
{
	fprintf(f, "\n(%lf, %lf, %lf)", cluster->center.x, cluster->center.y, cluster->center.z);
}

void writeNoResultToFile(char *fileName)
{
	FILE* f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("\nwriting to file was failed");
		return;
	}
	fprintf(f, "There is no solution!");
}
