#include "Calc_Picture.h"
#include <time.h>


// THE MAIN FUNCTION
/*
	Assamptions:
	1) 
*/
void Calc_Picture_Parallel(Params* params, Point* points, Result *result)
{

	int picID = result->PICTURE_ID;

	// Step 1:
	// Calc the new locations of the points due to time and speed-vector
	//example: for point which its center is (0,0,0) and speed (1,2,3) and picID is 2 when dt = 0.1, so:
	// next_center = (0 + 1*(0.1*2), 0 + 2*(0.1*2), 0 + 3*(0.1*2)) = (0.2, 0.4, 0.6)
	// runtime-complexity = O(N)
	Parallel_Calc_Points_Location(points, params, picID);

	// Step 2:
	// every cluster for start gain one point, which is its center
	// runtime-complexity = O(K)
	OMP_initClustersCenters(params, points, result);

	// Step 3:  
	// Goal: each point will know the index of the cluster in the array (which means: 0 to k-1) that its center is the closest near to her, 
	// by iterations that will calc wach iteration the new clusters' center
	// runtime-complexity = O(LIMIT*N*K)
	Parallel_set_optimal_clusters_centers(params, points, result);

	// At that point, each point knows her cluster' index

	// Step 4: 
	// Goal: each cluster will keep an array of indices of its points, for lowing the runtime complexity of Step 5
	// runtime-complexity = O(N)
	ClassifyClustersToPoints(params, points, result);

	// Step 5: 
	// Goal: find the biggest distance between two points in each cluster 
	// without Step 4: runtime-complexity = O(N^2)
	// With Step 4: runtime-complexity = O(N^2 / K)  [in AVERAGE case]
	// Due to the fact that: 2<=K<=N, there is an improvement of O(N^2) in runtime 
	Parallel_Calc_Clusters_Diameters(result, params, points);
	
	//step 6
	// calc the Q of the picture by the definition
	// runtime-complexity = O(K^2)
	CalcQ(params, result);

	// Step 7:
	// return the points to their previos locations (before )
	// runtime-complexity = O(N)
	int reconstructFuctor = (-picID);  // for return the location back
	Parallel_Calc_Points_Location(points, params, reconstructFuctor);
}

void Parallel_Calc_Points_Location(Point *points, Params *params, int picID)
{
	int numOfPoints_OMP = (int)(POINTS_LOCATIONS_OMP_PART * params->N);
	int numOfPoints_CUDA = params->N - numOfPoints_OMP;

	double totalDt = picID * params->dT;

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			if(numOfPoints_OMP > 0)
				OMP_CalcPointsLocation(totalDt, points, numOfPoints_OMP);
		}
		#pragma omp section
		{
			if(numOfPoints_CUDA > 0)
				CUDA_Calc_Points_Location(totalDt, &points[numOfPoints_OMP], numOfPoints_CUDA);
		}
	}
}

void Parallel_set_optimal_clusters_centers(Params* params, Point* points, Result* result)
{
	int isSystemStabilized = FALSE;
	int currentIteration = 0;

	while (currentIteration < params->LIMIT &&
		isSystemStabilized == FALSE)  // while there are more iterations to do AND the system is not stable, so:
	{
		// each point will remember its cluster which its center is the closest one for that point' location
		Parallel_Clasify_Points_To_Clusters(params, points, result);

		// update clusters' centers, and in the same time check if the system is stable or not
		isSystemStabilized = OMP_UpdateClustersCenters(params, points, result);
		
		// increament for get next iteration
		currentIteration++;
	}
}

void Parallel_Clasify_Points_To_Clusters(Params* params, Point* points, Result* result)
{
	// Set num of point for each thread
	int numOfPoints_OMP = (int)(params->N * CLUSIFICATION_OMP_PART);
	int numOfPoints_CUDA = params->N - numOfPoints_OMP;

	int is_cuda_success = TRUE;
	Point *OMP_start = points;
	Point *CUDA_start = &points[numOfPoints_OMP];
	
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			if (numOfPoints_OMP > 0)
				OMP_clasifyPointsToClusters(OMP_start, numOfPoints_OMP, result->clusters, params->K);
		}
		
		#pragma omp section
		{
			if (numOfPoints_CUDA > 0)
				CUDA_ClasifyPointsToClusters(CUDA_start, numOfPoints_CUDA, result->clusters, params->K);
		}
	}
}

/*
	the method splits their calculations by clusters. some clusters will be calc by OMP -
	and the rest by CUDA
*/
void Parallel_Calc_Clusters_Diameters(Result *result, Params* params, Point* points)
{
	// zeroed all the clusters
	int k;
	for (k = 0; k < params->K; k++)
		result->clusters[k].diameter = 0;

	int numOfClusters_OMP = (int)(params->K * DIAMETERS_OMP_PART);
	int numOfClusters_CUDA = params->K - numOfClusters_OMP;

	int is_cuda_success = TRUE;
	Cluster *OMP_start = result->clusters;
	Cluster *CUDA_start = result->clusters + numOfClusters_OMP;

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			
			if(numOfClusters_OMP > 0)
				OMP_CalcClustersDiameters(OMP_start, numOfClusters_OMP, points, params->N);
		}
		#pragma omp section
		{
			if (numOfClusters_CUDA > 0)
			{
				CUDA_CalcClustersDiameters(CUDA_start, numOfClusters_CUDA, points, params->N);
			}
		}
	}
}

void ClassifyClustersToPoints(Params *params, Point *points, Result *result)
{
	int j;
	for (j = 0; j < params->K; j++)
		result->clusters[j].numOfPoints = 0;

	int currentClusterIndex;
	Cluster *currentCluster;
	int nextIndex;
	int pointIndex;
	for (pointIndex = 0; pointIndex < params->N; pointIndex++)
	{
		currentClusterIndex = points[pointIndex].clusterID;
		currentCluster = &result->clusters[currentClusterIndex];
		nextIndex = currentCluster->numOfPoints;
		currentCluster->pointsIndices[nextIndex] = pointIndex;
		(currentCluster->numOfPoints)++;
	}
}

void CalcQ(Params* params, Result* result)
{
	/*
	calc Q for 3 clusters (as in the instructions, with some math calc):
	let's define: d = diameter of a cluster, D = distance between clusters
	so, the calculation of q will be:
	Q = ((d1+d2)/D12 + (d1+d3)/D13 + (d2+d3)/D23) / 6
	As we see: (sigma of (di + dj)/Dij) / (K * (K-1))   WHEN i<j
	*/

	double Q = 0;

	int i, j;
	for (i = 0; i < params->K; i++)
	{
		for (j = i + 1; j < params->K; j++) {
			// the method is the iteration. it gives: (di+dj)/Dij
			Q += calcQ_Helper(result, i, j);
		}
	}

	// by definition:
	result->Q = Q / (params->K * (params->K - 1));  
}

double calcQ_Helper(Result* result, int clusterIndex1, int clusterIndex2)
{
	// Goal: return: (di + dj)/Dij
	double di = result->clusters[clusterIndex1].diameter;
	double dj = result->clusters[clusterIndex2].diameter;
	double Dij = getDistanceBetween(result->clusters, clusterIndex1, clusterIndex2);
	return (di + dj) / Dij;
}


