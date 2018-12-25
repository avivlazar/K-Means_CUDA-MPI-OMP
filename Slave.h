#pragma once

#ifndef SLAVE_HEADER
#define SLAVE_HEADER


#include <mpi.h>
#include <omp.h>
#include "Calc_Picture.h"
#include <stdlib.h>


void SlaveProccess(Params *params, Point *points, int id);
static void searchForOptimalResult(Params* params, Point* points, Result* result, int id);
void SendMasterGoodResult(Result *result, Params *params, int id);
void SendMasterRequestForNextPicture(Result *result, int id);
int isQok(double Q, double QM);

#endif
