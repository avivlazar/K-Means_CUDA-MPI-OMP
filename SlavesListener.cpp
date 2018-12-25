#include "SlavesListener.h"

static int Picture_ID = -1; // global

/*
Purpose:
- send to slaves proper tags which start/continue/stop their work.
*/
// MAIN FUNCTION:
void activateSlavesListener(int numproc, AllData* allData)
{
	/* Introductions:
	Here, only one thread, with 4 main rules:
	1) sending a picture for every slave
	2) reciving from one slave each time a result
	3) check any time that naccessary if Master found a result.
	4) free the slaves from work
	*/

	if (numproc < 2)  // there isn't enough proccesses for a listener -> avoiding deadlock
		return;

	int numOfSlavesAtWork = 0;  // Assamption: all our slaves are free at start

	// Step 1: 
	// Sending one picture for every slave
	sendAllSlavesPicture(numproc, allData, &numOfSlavesAtWork);

	//Step 2:
	// recv pictures from slaves and check if they are good results.
	// if he recv a good one, he check and change (if neccassery) the result, and finish method.
	if(numOfSlavesAtWork > 0)
		recvResultsFromSlaves(allData, &numOfSlavesAtWork);

	// In this point, there are two options:
	// - a good result was found out, and was put in var 'result'
	// - there is no good result yet.

	// Step 3:
	// Waiting until all slaves finish calc the current picture, and free them
	// the master' thread check 
	if (numOfSlavesAtWork > 0)
		freeAllSlavesFromWork(allData, &numOfSlavesAtWork);

	// In this point: all the slaves don't work.
	// Situations:
	// - we finished all the pictures, and yet we didn't get solution.
	// - we found a good result, which was put in var 'result' in 'allData'.
}

// Step 1:
static void sendAllSlavesPicture(int numproc, AllData *allData, int *numOfSlavesAtWork)
{
	int slaveID;
	int pictureID;

	// send every slave one picture
	for (slaveID = 1; slaveID < numproc; slaveID++)
	{
		pictureID = getNextPictureID();
		
		// if the Master found a good result OR there are no pictures anymore -> release current slave
		// if it's so, we dont have to continue the searching
		if (allData->isMasterFoundResult 
			|| !isPictureLegal(pictureID, allData))
		{
			// if Master already found at short time good solution, OR There is no more pictures to send,
			// Master realese the slave.
			ReleaseSlave(slaveID);
		}
		else
		{
			// Else: there is no solution yet AND the picture id is OK, so:
			// - send the next picture to the next slave
			printf("\nslave=%d will calc picture=%d", slaveID, pictureID);
			fflush(stdout);
			MPI_Send(&pictureID, 1, MPI_INT, slaveID, PICTURE_TAG, MPI_COMM_WORLD);
			(*numOfSlavesAtWork)++;  // update num of slaves that work
			
		}
	}
}

/*
Assumptions:
- 'numOfSlavesAtWork' > 0
Proccess:
- getting next picture
- while there is not an optimal result yet AND there are pictures to send, so:
- recv results from slaves. if one of the result is good, store it in AllData,
realease this specific slave, and get out from the loop.
*/
static void recvResultsFromSlaves(AllData *allData, int *numOfSlavesAtWork)
{
	int pictureID = 0;
	int slaveID = 0;
	int tag = 0;
	int isThereNextPicture = TRUE;
	int isGoodResultFound = allData->isMasterFoundResult;
	Result *slavesListenerResult = &allData->slavesListenerResult;
	Vector* allCenters = (Vector*)calloc(allData->parameters->K, sizeof(Vector));
	MPI_Status status;

	while (!isGoodResultFound && isThereNextPicture)
	{
		// recv picture id
		MPI_Recv(&pictureID, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		(*numOfSlavesAtWork)--;  // update num of slaves which work
		slaveID = status.MPI_SOURCE;
		tag = status.MPI_TAG;
		fflush(stdout);
		
		if (tag == SUCCESS_TAG)
		{
			// send him the message: "I want your result"
			MPI_Send(&pictureID, 1, MPI_INT, slaveID, SEND_RESULT, MPI_COMM_WORLD);
			
			// set picture id:
			slavesListenerResult->PICTURE_ID = pictureID;
			
			// recv the hole result
			MPI_Recv(&(slavesListenerResult->Q), 1, MPI_DOUBLE, slaveID, BEST_Q_TAG, MPI_COMM_WORLD, &status);
			MPI_Datatype MPI_Vector = createVectorTypeF_MPI();
			
			// recv clusters' centers
			MPI_Recv(allCenters, allData->parameters->K, MPI_Vector, slaveID, CENTERS_TAG, MPI_COMM_WORLD, &status);

			int i;
			for (i = 0; i < allData->parameters->K; i++)
				slavesListenerResult->clusters[i].center = allCenters[i];
			
			ReleaseSlave(slaveID);
			isGoodResultFound = TRUE;
			allData->isSlavesListenerFoundResult = TRUE;
			
		}
		else  // else -> slave sent a bad result
			if (allData->isMasterFoundResult)  // if master already found a good result:
			{
				// maybe until the master' thread gain a result from the slave (a function which take time),
				// the master could find out a good result. if so, there is no reason to send the next picture to the slave.
				ReleaseSlave(slaveID);
				isGoodResultFound = TRUE;
			}
			else  // there is not good result yet (at all), do:
			{
				// Master ready to send the next picture to slave
				pictureID = getNextPictureID();
				if (isPictureLegal(pictureID, allData))
				{
					// send to the same slave the next picture
					//sendPictureID(slaveID, pictureID, numOfSlavesAtWork);
					MPI_Send(&pictureID, 1, MPI_INT, slaveID, PICTURE_TAG, MPI_COMM_WORLD);
					(*numOfSlavesAtWork)++;  // update num of slaves that work
					printf("\nslave=%d will calc picture=%d", slaveID, pictureID);
					fflush(stdout);
				}
				else   // there is not a good result and there are no more pictures to calc:
				{
					ReleaseSlave(slaveID);
					isThereNextPicture = FALSE;  // the loop will stop in next iteration
				}
			}
			
	}  // end of loop

	free(allCenters);
}



/*
Assumptions:
- OR Master finish to select all the pictures,
OR a good result was found (by Master itself - by help of the slaves),
OR the both reasons abouve come true.
- There are Slaves that still work.
- One or more of the slaves that still work, can provide a new optimal result.
Proccess:
- while there is at least one slave that still works, so:
- wait to recv from him his picture' id and an answer (if his result is good)
- if the result isgood -> recv from him the whole result (Q and clusters' information),
AND set the result.
- Release the slave, no matter what - due to first Assumption.
Purposes:
- Free all slaves from work.
- Clarify if there is an optimal solution.
*/
void freeAllSlavesFromWork(AllData *allData, int *numOfSlavesAtWork)
{
	int slaveID;
	int tag;
	int pictureID;
	Result* slavesListenerResult = &allData->slavesListenerResult;
	MPI_Status status;
	//int isResultFound = allData->isMasterFoundResult | allData->isSlavesListenerFoundResult;

	while (*numOfSlavesAtWork > 0)
	{
		// recv from the next slave its work.as well - udate the number of slaves at 
		RecvPictureID_FromSlave(&pictureID, &slaveID, &tag, numOfSlavesAtWork);

		if (tag == SUCCESS_TAG)
		{
			// if slaves' listener didn't get a good result yet, or get good result, but there is a better one, so:
			if (pictureID < slavesListenerResult->PICTURE_ID)
			{
				MPI_Send(&pictureID, 1, MPI_INT, slaveID, SEND_RESULT, MPI_COMM_WORLD);
				
				// set picture id
				slavesListenerResult->PICTURE_ID = pictureID;

				// recv Q value
				MPI_Recv(&(slavesListenerResult->Q), 1, MPI_DOUBLE, slaveID, BEST_Q_TAG, MPI_COMM_WORLD, &status);

				MPI_Datatype MPI_Vector = createVectorTypeF_MPI();
				Vector* allCenters = (Vector*)calloc(allData->parameters->K, sizeof(Vector));
				MPI_Recv(allCenters, allData->parameters->K, MPI_Vector, slaveID, CENTERS_TAG, MPI_COMM_WORLD, &status);

				int i;
				for (i = 0; i < allData->parameters->K; i++)
					slavesListenerResult->clusters[i].center = allCenters[i];

				allData->isSlavesListenerFoundResult = TRUE;
			}
			else
				MPI_Send(&pictureID, 1, MPI_INT, slaveID, DONT_SEND_RESULT, MPI_COMM_WORLD);
		}
		// else: the solution of the slave isn't good, so we don't care.

		ReleaseSlave(slaveID);
	}
}

static void ReleaseSlave(int slaveID)
{
	int tmp = -1;
	MPI_Send(&tmp, 1, MPI_INT, slaveID, RELEASE_TAG, MPI_COMM_WORLD);
}

void RecvPictureID_FromSlave(int *pictureID, int *slaveID, int *picTag, int *numOfSlavesAtWork)
{
	MPI_Status status;
	// recv the picture id: (and tag)
	MPI_Recv(pictureID, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	(*numOfSlavesAtWork)--;  // update num of slaves which work
	*slaveID = status.MPI_SOURCE;
	*picTag = status.MPI_TAG;
}

/*
Assamption: more than one thread (in this program - at most two threads)
may be here in the same time.
each thread need to change the value of the next picture.
*/
int getNextPictureID()
{
		#pragma omp atomic 
		Picture_ID++;
		return Picture_ID;
}

void swapResults(Result *r1, Result *r2)
{
	Result temp = *r1;
	*r1 = *r2;
	*r2 = *r1;
}

int isPictureLegal(int pictureID, AllData* allData)
{
	if (pictureID < getNumOfPictures(allData->parameters))
		return TRUE;
	return FALSE;
}

