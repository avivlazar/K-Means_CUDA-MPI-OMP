
#pragma once

#ifndef DATA_STRUCTS_HEADER
#define DATA_STRUCTS_HEADER

#include <stdlib.h>

#define NO_SOLUTION_YET -1
#define TRUE 1
#define FALSE 0

struct VectorStruct {
	double x;
	double y;
	double z;
} typedef Vector;

struct PointStruct {
	int clusterID;
	Vector location; // the loction' vector of the point in area
	Vector speed; // the speed' vector of the point in area
} typedef Point;

struct ClusterStruct {
	Vector center;
	double diameter;
	int *pointsIndices;
	int numOfPoints;
} typedef Cluster;

struct ResulStruct {
	int PICTURE_ID;
	double Q = NO_SOLUTION_YET;
	Cluster* clusters;
}typedef Result;

struct ParametersStruct {
	int N; // num Of Points
	int K; // num of clusters
	int LIMIT;  // limit of iterations for finding clusters centers
	double QM;  // the result of the picture
	double T; // the end time interval
	double dT; // the space time between one picture to another 
}typedef Params;

struct AllDataStruct {
	Params *parameters;
	Point *points;
	Result slavesListenerResult;
	Result masterResult;
	int isSlavesListenerFoundResult;
	int isMasterFoundResult;
}typedef AllData;


#endif
