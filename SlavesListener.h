#pragma once

#ifndef MASTER_SEARCH_RESULT_BY_SLAVES
#define MASTER_SEARCH_RESULT_BY_SLAVES

#include "Structs.h"
#include "Calc_Picture.h"
#include "Structs_MPI.h"
#include <omp.h>

// MAIN FUNCTION
void activateSlavesListener(int numproc, AllData* allData);

static void sendAllSlavesPicture(int numproc, AllData *allData, int *numOfSlavesAtWork);
static void recvResultsFromSlaves(AllData *allData, int *numOfSlavesAtWork);
void freeAllSlavesFromWork(AllData *allData, int *numOfSlavesAtWork);

// USEFUL METHODS:
static void ReleaseSlave(int slaveID);
void RecvPictureID_FromSlave(int* pictureID, int *slaveID, int *picTag, int *numOfSlavesAtWork);
int getNextPictureID();
void swapResults(Result *r1, Result *r2);
int isPictureLegal(int pictureID, AllData* allData);


#endif // !MASTER_SEARCH_RESULT_BY_SLAVES
