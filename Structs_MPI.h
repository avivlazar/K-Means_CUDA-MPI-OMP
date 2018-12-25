#pragma once
#pragma once

#ifndef DATA_STRUCTS_MPI
#define DATA_STRUCTS_MPI

#include <mpi.h>
#include "Structs.h"
#include <stdlib.h>

#define SEND_RESULT 1
#define DONT_SEND_RESULT 0

#define MASTER 0
#define POINTS_TAG 100
#define PICTURE_TAG 200
#define NEXT_POINT_TAG 300
#define SUCCESS_TAG 400
#define RELEASE_TAG 500
#define PARAMS_TAG 600
#define BEST_Q_TAG 700
#define CENTERS_TAG 800
#define DEFAULT_TAG 1000

MPI_Datatype createVectorTypeF_MPI();
MPI_Datatype createPointType_MPI();
MPI_Datatype createParamsType_MPI();

#endif // !DATA_STRUCTS_MPI

