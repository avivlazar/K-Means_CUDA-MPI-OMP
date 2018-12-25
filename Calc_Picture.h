#pragma once

#ifndef PALC_PICTURE_HEADER
#define PALC_PICTURE_HEADER

#include "OMP_Calc_Picture.h"
#include "CUDA_Calc_Picture.h"


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define POINTS_LOCATIONS_OMP_PART 0.1  // between 0 to 1
#define CLUSIFICATION_OMP_PART 0.1  // 0 til 1. The rest: CUDA
#define DIAMETERS_OMP_PART 0.5 // 0 til 1. The rest: CUDA

void Calc_Picture_Parallel(Params* params, Point* points, Result *result);
void Parallel_Calc_Points_Location(Point *points, Params *params, int picID);
double getDistanceBetween(Cluster* clusters, int i, int j);
int isQok(double Q, double QM);
int getNumOfPictures(Params* params);
void Parallel_set_optimal_clusters_centers(Params* params, Point* points, Result* result);
void Parallel_Clasify_Points_To_Clusters(Params* params, Point* points, Result* result);
void ClassifyClustersToPoints(Params *params, Point *points, Result *result);
void Parallel_Calc_Clusters_Diameters(Result *result, Params* params, Point* points);
void CalcQ(Params* params, Result* result);
double calcQ_Helper(Result* result, int i, int j);


#endif   //  CALC_PICTURE_HEADER
