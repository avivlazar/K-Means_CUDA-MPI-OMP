#pragma once

#ifndef OMP_CALC_PICTURE
#define OMP_CALC_PICTURE

#include "Structs.h"
#include <math.h>
#include <omp.h>


void OMP_CalcPointsLocation(double totalDt, Point * points, int numOfPoints);
void OMP_initClustersCenters(Params* params, Point* points, Result* result);

// Step 4
int getNumOfPictures(Params* params);
void OMP_clasifyPointsToClusters(Point* points, int numOfPoints, Cluster *clusters, int numOfClusters);
void OMP_CalcSumsOfLocations(Params * params, Point * points, Vector * sumsOfLocations);
void CalcNumOfPointsPerCluster(Params * params, Point * points, int * numOfPointsPerCluster);  
int OMP_UpdateClustersCenters(Params* params, Point* points, Result* result);
void OMP_CalcClustersDiameters(Cluster *clusters, int numOfClusters, Point * points, int numOfPoints);


// USEFUL METHODS:
double getVectorLength(Vector* vec);
double getDistanceBetweenVectors(Vector* v1, Vector* v2);
double getDistanceBetweenPoints(Point *p1, Point *p2);
static double getDistanceBetween(Point* point, Cluster* cluster);
void OMP_Init_Threads_Bounderies_To_Calc_Pragma(int start, int end, int *myStart, int *myEnd);
double getDistanceBetween(Cluster* clusters, int i, int j);
double getMax(double *array, int size);
double getDistanceBetweenVectors(Vector* v1, Vector* v2);


#endif // !OMP_CALC_PICTURE

