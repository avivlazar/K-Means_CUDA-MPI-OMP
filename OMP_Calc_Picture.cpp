#include "OMP_Calc_Picture.h"
#include <stdio.h>
void OMP_CalcPointsLocation(double totalDt, Point * points, int numOfPoints)
{
	#pragma omp parallel
	{
		int myStart, myEnd;
		OMP_Init_Threads_Bounderies_To_Calc_Pragma(0, numOfPoints, &myStart, &myEnd);

		int i;
		for (i = myStart; i < myEnd; i++)
		{
			points[i].location.x += (points[i].speed.x * totalDt);
			points[i].location.y += (points[i].speed.y * totalDt);
			points[i].location.z += (points[i].speed.z * totalDt);
		}
	}
}


void OMP_initClustersCenters(Params* params, Point* points, Result* result)
{
	#pragma omp parallel
	{
		int i;
		Cluster *currentCluster;
		int myStart, myEnd;
		OMP_Init_Threads_Bounderies_To_Calc_Pragma(0, params->K, &myStart, &myEnd);

		for (i = myStart; i < myEnd; i++)
		{
			currentCluster = &(result->clusters[i]);

			// as part of the init - cluster 'i' gain only one point, and that point will be his center
			// that's happend for promising each cluster have at least one point in him
			currentCluster->center = points[i].location;
			points[i].clusterID = i;
			currentCluster->numOfPoints = 1;
			currentCluster->pointsIndices[0] = i;
		}
	}
}


void OMP_clasifyPointsToClusters(Point* points, int numOfPoints, Cluster *clusters, int numOfClusters)
{	
	#pragma omp parallel
	{
		int minIndex;  // the index of the closest cluster in the array to a point
		double minDistance;  // the distance between the closest cluster in the array to a point
		int myStart, myEnd;
		Point* currentPoint;
		Cluster *currentCluster;
		OMP_Init_Threads_Bounderies_To_Calc_Pragma(0, numOfPoints, &myStart, &myEnd);
		
		int i;
		for (i = myStart; i < myEnd; i++)
		{ 
			minIndex = 0;  // the cluster' index which its center is the nearest one to the current point
			currentPoint = &points[i];
			currentCluster = &clusters[minIndex];
			minDistance = getDistanceBetweenVectors(&currentPoint->location, &currentCluster->center);  // remember the minimal distance
			double currentDistance;
			int j;
			for (j = 1; j < numOfClusters; j++)
			{
				currentCluster = &clusters[j];
				currentDistance = getDistanceBetweenVectors(&currentPoint->location, &currentCluster->center);
				if (currentDistance < minDistance)
				{
					minDistance = currentDistance;
					minIndex = j;
				}
			}

			// At that point: minIndex got the cluster' index which its center is the closest one tho current point' location

			// set the cluaster index of the point:
			currentPoint->clusterID = minIndex;
		}
	}
}


void OMP_CalcSumsOfLocations(Params * params, Point * points, Vector * sumsOfLocations)
{
	int i;
	Point *currentPoint;
	int clusterIndex;
	for (i = 0; i < params->N; i++)
	{
		currentPoint = &points[i];
		clusterIndex = currentPoint->clusterID;

		sumsOfLocations[clusterIndex].x += currentPoint->location.x;
		sumsOfLocations[clusterIndex].y += currentPoint->location.y;
		sumsOfLocations[clusterIndex].z += currentPoint->location.z;
	}
}


void CalcNumOfPointsPerCluster(Params * params, Point * points,
	int * numOfPointsPerCluster)
{
	int i, clusterOfPoint;
	for (i = 0; i < params->N; i++) 
	{
		clusterOfPoint = points[i].clusterID;
		numOfPointsPerCluster[clusterOfPoint]++;
	}
}

// calc and update the clusters' centers. 
// return: 1 if the system is stable, and 0 otherwise. 
int OMP_UpdateClustersCenters(Params* params, Point* points, Result* result)
{
	Vector *sumsOfLocations = (Vector*)calloc(params->K, sizeof(Vector));  // array of zeroed vectors
	int *numOfPointsPerCluster = (int*)calloc(params->K, sizeof(int));  // array of zeroed int

	OMP_CalcSumsOfLocations(params, points, sumsOfLocations);

	CalcNumOfPointsPerCluster(params, points, numOfPointsPerCluster);
	
	Vector* currentClusterCenter;
	int isSystemStabled = TRUE;  // for start
	double x, y, z;
	int i;
	for (i = 0; i < params->K; i++)
	{
		currentClusterCenter = &(result->clusters[i].center);

		// the new cluster' center
		x = sumsOfLocations[i].x / numOfPointsPerCluster[i];
		y = sumsOfLocations[i].y / numOfPointsPerCluster[i];
		z = sumsOfLocations[i].z / numOfPointsPerCluster[i];

		// if the new center is d
		if (x != currentClusterCenter->x || y != currentClusterCenter->y || z != currentClusterCenter->z)
		{
			// set the new center in current cluster
			currentClusterCenter->x = x;
			currentClusterCenter->y = y;
			currentClusterCenter->z = z;

			isSystemStabled = FALSE;
		}
	}

	free(sumsOfLocations);
	free(numOfPointsPerCluster);

	return isSystemStabled;
}


/*
	for each cluster Ci, the method calc its diameter by calc each two points (P1, P2) - 
	which is contained in cluster Ci
*/
void OMP_CalcClustersDiameters(Cluster *clusters, int numOfClusters, Point * points, int numOfPoints)
{
	
	Cluster *currentCluster;
	int numOfPointsInCurrentCluster;
	// array with zeroes - which store the result of each thread:
	int numOfResults = numOfPoints;
	double* results = (double*)calloc(numOfResults, sizeof(double));  // will contain all posibilities to cluster' diameter

	int i;
	for (i = 0; i < numOfClusters; i++)
	{
		currentCluster = &clusters[i];
		numOfPointsInCurrentCluster = currentCluster->numOfPoints;

		#pragma omp parallel
		{
			int j, m;
			int myStart, myEnd;
			int pointIndex1, index2;
			Point *point1, *point2;
			double currentDistance = 0;
			OMP_Init_Threads_Bounderies_To_Calc_Pragma(0, numOfPointsInCurrentCluster, &myStart, &myEnd);

			for (j = myStart; j < myEnd; j++)
			{
				pointIndex1 = currentCluster->pointsIndices[j];
				point1 = &(points[pointIndex1]);

				for (m = j + 1; m < numOfPointsInCurrentCluster; m++)
				{
					index2 = currentCluster->pointsIndices[m];
					point2 = &(points[index2]);
					currentDistance = getDistanceBetweenPoints(point1, point2);
					if (results[j] < currentDistance)
						results[j] = currentDistance;
				}
			}
		}

		currentCluster->diameter = getMax(results, numOfResults);

		// we have to reset the result array to zeroes for the next cluster' calc
		int q;
		for (q = 0; q < numOfPointsInCurrentCluster; q++)
			results[q] = 0;
	}
	free(results);
}

double getMax(double *array, int size)
{
	double maxResult = array[0];
	int i; 
	for (i = 1; i < size; i++)
	{
		if (maxResult < array[i])
			maxResult = array[i];
	}
	return maxResult;
}

// give every thread which come inside its own start and end by their id.
// example: start = 0, end = 10, num of threads = 5
// thread 1: 0 -> 2,    thread 1: 2 -> 4,    thread 2: 4 -> 6,    thread 3: 6 -> 8,    thread 4: 8 -> 10   [threadStart -> threadEnd]
void OMP_Init_Threads_Bounderies_To_Calc_Pragma(int start, int end, int *threadStart, int *threadEnd)
{
	int numOfThreds = omp_get_num_threads();
	int myid = omp_get_thread_num();
	int range = (int)(((end - start) + numOfThreds - 1) / numOfThreds);
	int temp = start;
	
	*threadStart = start + (myid * range);
	*threadEnd = *threadStart + range;
	if (end < *threadEnd)  // for the last thread
		*threadEnd = end;
}

double getDistanceBetweenVectors(Vector* v1, Vector* v2)
{
	Vector distanceVec;
	distanceVec.x = v2->x - v1->x;
	distanceVec.y = v2->y - v1->y;
	distanceVec.z = v2->z - v1->z;
	return getVectorLength(&distanceVec);
}

double getVectorLength(Vector* vec)
{
	return sqrt(pow(vec->x, 2) + pow(vec->y, 2) + pow(vec->z, 2));
}

double getDistanceBetweenPoints(Point *p1, Point *p2)
{
	return getDistanceBetweenVectors(&(p1->location), &(p2->location));
}

static double getDistanceBetween(Point* point, Cluster* cluster)
{
	return getDistanceBetweenVectors(&(point->location), &(cluster->center));
}

int getNumOfPictures(Params* params)
{
	return (int)((params->T + params->dT) / params->dT);
}

double getDistanceBetween(Cluster* clusters, int i, int j)
{
	Vector ci = clusters[i].center;  // center of cluster i
	Vector cj = clusters[j].center;  // center of cluster j
	return getDistanceBetweenVectors(&ci, &cj);
}