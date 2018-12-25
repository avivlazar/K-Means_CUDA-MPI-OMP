
#include "Slave.h"
#include "Structs_MPI.h"

void SlaveProccess(Params *parameters, Point *points, int id)
{
	int numOfClusters = parameters->K;
	int numOfPoints = parameters->N;
	Result result;

	// Create slave' result struct 
	result.clusters = (Cluster*)calloc(numOfClusters, sizeof(Cluster));
	int i;
	for (i = 0; i < parameters->K; i++)
	{
		result.clusters[i].pointsIndices = (int*)calloc(numOfPoints, sizeof(int));
	}

	// main method: find an optimal resuult - and send to master
	searchForOptimalResult(parameters, points, &result, id);

	// free result' memory
	int j;
	for (j = 0; j < numOfClusters; j++)
	{
		free(result.clusters[j].pointsIndices);
	}
	free(result.clusters);
}


/*
	---SEARCH FOR OPTIMAL RESULT---
Proccess:
- recv from master a picture' id
- while slave didn't get a RELEASE TAG from Master, do:
- calc the picture' Q
- if Q is good -> the slave send Master the result.
- else -> the result is not good, so: 
	-the slave send Master request to the next picture.
- slave wait to recv an answer from the master what to do (by a tag)
Purpose:
- To search for an optimal result
*/
static void searchForOptimalResult(Params* params, Point* points, Result* result, int id)
{
	int isReleasedTag;  // 
	MPI_Status status;

	// Step 1:
	// recv picture id into 'rasult':
	MPI_Recv(&(result->PICTURE_ID), 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	isReleasedTag = status.MPI_TAG;

	while (isReleasedTag != RELEASE_TAG)
	{
		// Calc the Q and the clusters' centers, and write in result
		Calc_Picture_Parallel(params, points, result);

		// if the Q of the picture is good, so:
		if (isQok(result->Q, params->QM))
			SendMasterGoodResult(result, params, id);
		else
			SendMasterRequestForNextPicture(result, id);

		MPI_Recv(&(result->PICTURE_ID), 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		isReleasedTag = status.MPI_TAG;
	}
}

/*
	Proccess:
	1) Slave send the ID ot the picture + a success tag
	2) Master recv - and expects from the slave to send the hole result.
	3) Slave send the Q (which is good one) and the clusters. (without the pointsIndices arrays)
*/
void SendMasterGoodResult(Result *result, Params *params, int id)
{
	// send the picture id to master
	MPI_Send(&(result->PICTURE_ID), 1, MPI_INT, MASTER, SUCCESS_TAG, MPI_COMM_WORLD);
	
	// Recv master' preference
	int tmp;
	MPI_Status status;

	MPI_Recv(&tmp, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	int tag = status.MPI_TAG;

	// if master want my result, so:
	if (tag == SEND_RESULT)
	{
		// send q result
		MPI_Send(&(result->Q), 1, MPI_DOUBLE, MASTER, BEST_Q_TAG, MPI_COMM_WORLD);
		
		// replace all centers to a special vector' array 
		Vector* allCenters = (Vector*)calloc(params->K, sizeof(Vector));
		int i;
		for (i = 0; i < params->K; i++)
			allCenters[i] = result->clusters[i].center;

		// send array of clusters
		MPI_Datatype MPI_Vector = createVectorTypeF_MPI();
		MPI_Send(allCenters, params->K, MPI_Vector, MASTER, CENTERS_TAG, MPI_COMM_WORLD);

		// free the centers array
		free(allCenters);
	}
}

void SendMasterRequestForNextPicture(Result *result, int id)
{
	MPI_Send(&(result->PICTURE_ID), 1, MPI_INT, MASTER, PICTURE_TAG, MPI_COMM_WORLD);
}

int isQok(double Q, double QM)
{
	if (Q != NO_SOLUTION_YET && Q < QM)
	{
		return TRUE;
	}
	return FALSE;
}
