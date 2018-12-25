#pragma once
#ifndef CALC_PICTURE_CUDA
#define CALC_PICTURE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Structs.h"

// MAIN FUNCTIONS' ANNOUNCMENTS OF CPU:
void CUDA_Calc_Points_Location(double totalDt, Point * points, int numOfPoints);
__global__ void calcSinglePointLocation_CUDA(double totalDt, Point * points, int numOfPoints);


void CUDA_ClasifyPointsToClusters(Point* points, int numOfPoints, Cluster *clusters, int numOfClusters);
__global__ void clusifySinglePointToCluster_CUDA(Vector *centers, int numOfCenters, Point *points, int numOfPoints);
__device__ int get_cluster_ID_which_give_minimal_distance_with(Point *currentPoint, Vector *centers, int numOfCenters);

void CUDA_CalcClustersDiameters(Cluster* clusters, int numOfClusters, Point * points, int numOfPoints);
__global__ void Set_Dist_Between_Points(int numOfIndices, int *clusterIndicesPointers, Point *points, double *results, int currentPointIndex);

__device__ double getDistanceBetween(Point* p1, Point* p2);
__device__ double getDistanceBetween(Vector *center, Point *point);
__device__ double getDistanceBetween(Vector *vec1, Vector *vec2);


int getNumOfBlocks(int numOfPoints);
double getMaxValue(double *array, int size);
#endif // !CALC_PICTURE_CUDA
