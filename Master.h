#pragma once

#ifndef MASTER_HEADER
#define MASTER_HEADER

#include <mpi.h>

#include "Calc_Picture.h"
#include "SlavesListener.h"


// MAIN FUNCTION:
void MasterProccess(int numproc, Params *params, Point *points, char *outputFileName);

// Step 1:
void initMemoryResultsForAllData(AllData *allData);

//Step 3:
void Search_Result(int numproc, AllData *allData);

static void ReleaseSlave(int slaveID);

void Master_Search(AllData *allData);
static void printPicture(Result *result, int numOfClusters);
static void printClusters(Cluster *clusters, int numOfClusters);
static void printVector(Vector *vector);
int isQok(Result *myResult, AllData* allData);
void writeSuccessResultToFile(char *fileName, Result *result, Params *params);
void writeNoResultToFile(char *fileName);
void printClusterToFile(FILE* f, Cluster *cluster);

#endif
