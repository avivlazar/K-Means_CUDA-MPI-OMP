
#include "CUDA_Calc_Picture.h"
#include <math.h>
#define NUM_OF_THREADS 512

void CUDA_Calc_Points_Location(double totalDt, Point * points, int numOfPoints)
{
	Point *dev_points;
	int pointsSize = numOfPoints * sizeof(Point);
	cudaError status;

	// Set Device:
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}
		
	
	// Malloc Points
	status = cudaMalloc((void**)&dev_points, pointsSize);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Copy Points To Device
	status = cudaMemcpy(dev_points, points, pointsSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Calc num of blocks
	int numOfBlocks = getNumOfBlocks(numOfPoints);

	// Kernel
	calcSinglePointLocation_CUDA << < numOfBlocks, NUM_OF_THREADS >> > (totalDt, dev_points, numOfPoints);

	// copy points back to host
	status = cudaMemcpy(points, dev_points, pointsSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// free points from device
	status = cudaFree(dev_points);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}
}

__global__ void calcSinglePointLocation_CUDA(double totalDt, Point * points, int numOfPoints)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x);  // = threadIndex
	if (i < numOfPoints)
	{
		points[i].location.x += points[i].speed.x * totalDt;
		points[i].location.y += points[i].speed.y * totalDt;
		points[i].location.z += points[i].speed.z * totalDt;
	}
}


void CUDA_ClasifyPointsToClusters(Point* points, int numOfPoints, Cluster *clusters, int numOfClusters)
{
	Point *dev_points;
	Vector *dev_centers;
	cudaError status;

	Vector *centers = (Vector*)calloc(numOfClusters, sizeof(Vector));
	int pointsSize = numOfPoints * sizeof(Point);
	int centersSize = numOfClusters * sizeof(Vector);

	// copy the centers of the clusters in the cpu
	int i;
	for (i = 0; i < numOfClusters; i++)
		centers[i] = clusters[i].center;

	// reset device
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// malloc points in GPU
	status = cudaMalloc((void**)&dev_points, pointsSize);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// malloc centers in GPU
	status = cudaMalloc((void**)&dev_centers, centersSize);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// copy points to GPU
	status = cudaMemcpy(dev_points, points, pointsSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}
	
	// copy clusters' centers to GPU
	status = cudaMemcpy(dev_centers, centers, centersSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Calc num of blocks
	int numOfBlocks = getNumOfBlocks(numOfPoints);

	// Kernel
	clusifySinglePointToCluster_CUDA << < numOfBlocks, NUM_OF_THREADS >> > (dev_centers, numOfClusters, dev_points, numOfPoints);

	// copy points to host
	status = cudaMemcpy(points, dev_points, pointsSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// free clusters in GPU
	status = cudaFree(dev_centers);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// free points in GPU
	status = cudaFree(dev_points);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}
	
	// free centers array in GPU
	free(centers);
}

__global__ void clusifySinglePointToCluster_CUDA(Vector *centers, int numOfCenters, Point *points, int numOfPoints)
{
	int threadIndex = (threadIdx.x + (blockIdx.x * blockDim.x));
	
	if (threadIndex < numOfPoints)
	{
		Point *currentPoint = points + threadIndex;  
															// runtime complexity: O(numOfClusters)
		currentPoint->clusterID = get_cluster_ID_which_give_minimal_distance_with(currentPoint, centers, numOfCenters);
	}
}

__device__ int get_cluster_ID_which_give_minimal_distance_with(Point *point, Vector *centers, int numOfCenters)
{
	int minIndex = 0;
	double minDist = getDistanceBetween(&centers[minIndex], point);
	double currentDist;
	Vector *currentCenter;
	int i;
	for (i = 1; i < numOfCenters; i++)
	{
		currentCenter = &centers[i];
		currentDist = getDistanceBetween(currentCenter, point);
		if (currentDist < minDist)
		{
			minDist = currentDist;
			minIndex = i;
		}
	}
	return minIndex;
}


void CUDA_CalcClustersDiameters(Cluster* clusters, int numOfClusters, Point * points, int numOfPoints)
{
	Point *dev_points;
	double *dev_results;
	int *dev_indices;
	cudaError status;

	int numOfResults = numOfPoints;
	int numOfIndices = numOfPoints;
	double *results = (double*)calloc(numOfResults, sizeof(double));
	double *reslt = (double*)calloc(numOfResults, sizeof(double));
	int *indices = (int*)calloc(numOfIndices, sizeof(double));

	// Set Device:
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Malloc Points in GPU
	status = cudaMalloc((void**)&dev_points, numOfPoints * sizeof(Point));
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Malloc results in GPU
	status = cudaMalloc((void**)&dev_results, numOfResults * sizeof(double));
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Malloc indices' pooints of the cluster - in GPU
	status = cudaMalloc((void**)&dev_indices, numOfIndices * sizeof(int));
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Copy points To Device
	status = cudaMemcpy(dev_points, points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// Calc num of blocks
	
	int i, j;
	for (i = 0; i < numOfClusters; i++)
	{
		// Copy the pointts' indices to idices array
		for (int t = 0; t < clusters[i].numOfPoints; t++)
			indices[t] = clusters[i].pointsIndices[t];

		// Copy points' indices To Device
		status = cudaMemcpy(dev_indices, indices, numOfIndices * sizeof(int), cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			printf("\ncuda Error: %d", status);
			return;
		}

		
		int currentClusterNumOfIndices = clusters[i].numOfPoints;
		int numOfBlocks = getNumOfBlocks(currentClusterNumOfIndices);

		for (j = 0; j < currentClusterNumOfIndices; j++)
		{
			/*
			for each point we calc all distance that may be
			every call to kernel fo Pj it calc dist: Pj->P0, Pj->P1 ...  ,  = result
			reslt[j] = max(result) int etaration j
			*/
			int currentPointIndex = clusters[i].pointsIndices[j];
			
			//Kernel: fill 'results' arry with: Pj->P0, Pj->P1 ... 
			Set_Dist_Between_Points << < numOfBlocks, NUM_OF_THREADS >> > (currentClusterNumOfIndices, dev_indices, dev_points, dev_results, currentPointIndex);
			cudaDeviceSynchronize();
			// Memcopy results to host - with real diameter
			status = cudaMemcpy(results, dev_results, numOfResults * sizeof(double), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				printf("\ncuda Error: %d", status);
				return;
			}
			
			// set the max result as an optional diameter (with pointIndex j) - by definition
			reslt[j] = getMaxValue(results, numOfResults);
		}

		// set the max result as a diameter - by definition
		clusters[i].diameter = getMaxValue(reslt, numOfResults);

		// set back to zero - for next iteration
		for (int k = 0; k < numOfResults; k++)
			reslt[k] = 0;
	}

	// free All points in GPU
	status = cudaFree(dev_points);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// free results in GPU
	status = cudaFree(dev_results);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// free points' indices
	status = cudaFree(dev_indices);
	if (status != cudaSuccess)
	{
		printf("\ncuda Error: %d", status);
		return;
	}

	// free results of CPU
	free(results);
	free(reslt);
}

double getMaxValue(double *array, int size)
{
	double max = array[0];
	int i;
	for (i = 1; i < size; i++)
		if (max < array[i])
			max = array[i];
	return max;
}


__global__ void Set_Dist_Between_Points(int numOfIndices, int *clusterIndicesPointers, Point *points, double *results, int currentPointIndex)
{
	int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIndex < numOfIndices)
	{
		int index1 = currentPointIndex;
		int index2 = clusterIndicesPointers[threadIndex];
		results[threadIndex] = getDistanceBetween(&points[index1], &points[index2]);
	}
}



__device__ double getDistanceBetween(Point* p1, Point* p2)
{
	return getDistanceBetween(&(p1->location), &(p2->location));
}


__device__ double getDistanceBetween(Vector *center, Point *point)
{
	Vector *vec1 = center;
	Vector *vec2 = &(point->location);
	return getDistanceBetween(vec1, vec2);
}

__device__ double getDistanceBetween(Vector *vec1, Vector *vec2)
{
	double x = vec1->x - vec2->x;
	double y = vec1->y - vec2->y;
	double z = vec1->z - vec2->z;
	return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}

int getNumOfBlocks(int numOfPoints)
{
	return (int)((numOfPoints + NUM_OF_THREADS - 1) / NUM_OF_THREADS);
}