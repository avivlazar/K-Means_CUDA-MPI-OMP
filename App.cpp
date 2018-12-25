#include "Slave.h"
#include "Master.h"
#include "mpi.h"
#pragma warning(disable: 4996)

#define FOPEN_FAILED 0
#define FOPEN_SUCCESS 1


static void MASTER_SLAVE_Search(int id, int numproc, Params *params, Point *points, char *outputFileName);
void init_mpi(int argc, char *argv[], int *id, int *numprocs);
void send_Params_And_Points(Params *parameters, Point *points);
void readParams(Params *params, FILE *f);
void Master_readPoints(Point *points, int numOfPoints, FILE* f);
int Open_File(const char *fileName, FILE* f);


int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		printf("\nFiles for reading or writing are missing!");
		return 1;
	}
	const char *fileName = argv[1];
	int id;
	int numprocs;
	Params parameters;
	Point *points = NULL;
	int exit = FALSE;
	FILE *f;

	init_mpi(argc, argv, &id, &numprocs);

	if (id == 0)
	{
		f = fopen(fileName, "r");
		if(f == NULL)
			exit = TRUE;
	}

	MPI_Bcast(&exit, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (exit == TRUE)
		goto Exit;

	if (id == 0)
		readParams(&parameters, f);
	if (parameters.K > parameters.N)
	{
		printf("\nCan't be more clusters than points.\nPlease check your input.\n");
		goto Exit;
	}
	if (parameters.dT > parameters.T)
	{
		printf("\nDt can't be bigger than T.\nPlease check your input.\n");
		goto Exit;
	}
	

	MPI_Datatype MPI_Params = createParamsType_MPI();
	MPI_Bcast(&parameters, 1, MPI_Params, MASTER, MPI_COMM_WORLD);
	
	points = (Point*)calloc(parameters.N, sizeof(Point));

	if (id == 0)
		Master_readPoints(points, parameters.N, f);

	MPI_Datatype MPI_Point = createPointType_MPI();
	MPI_Bcast(points, parameters.N, MPI_Point, MASTER, MPI_COMM_WORLD);

	double start = MPI_Wtime();
	MASTER_SLAVE_Search(id, numprocs, &parameters, points, argv[2]);
	double end = MPI_Wtime();
	double totalTime = end - start;
	printf("\nTotal time is: %lf (sec)", totalTime);
	fflush(stdout);

	if (id == MASTER)
	{
		printf("\nMaster exit from program\n");
		free(points);
	}
	else
		printf("\nSlave=%d exit from program\n", id);
	fflush(stdout);

Exit:
	MPI_Finalize();
	return 0;
}

/*
	Init the MPI to Master and Slaves
*/
void init_mpi(int argc, char *argv[], int *id, int *numprocs)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, id);
	MPI_Comm_size(MPI_COMM_WORLD, numprocs);

	// For promising using parallelism in sections, we use this method:
	omp_set_nested(1);
	omp_set_dynamic(0);

	// Set max threads
	int maxThreads = omp_get_max_threads();
	omp_set_num_threads(maxThreads);
}

void send_Params_And_Points(Params *parameters, Point *points)
{
	MPI_Datatype MPI_Params = createParamsType_MPI();
	MPI_Datatype MPI_Point = createPointType_MPI();

	MPI_Bcast(&parameters, 1, MPI_Params, MASTER, MPI_COMM_WORLD);
	// Send all slaves the points:
	int numOfPoints = parameters->N;
	MPI_Bcast(points, numOfPoints, MPI_Point, MASTER, MPI_COMM_WORLD);
}

int Open_File(const char *fileName, FILE* f)
{
	f = fopen(fileName, "r");  // file pointer
							   // check if file is good 
	if (f == NULL)
	{
		printf("\n Failed opening the file. Exiting!\n");
		return FOPEN_FAILED;
	}
	return FOPEN_SUCCESS;
}

/*
	Scan all the parameters for strart: num of points, num of clusters, time, delta in time, LIMIT and QM 
*/
void readParams(Params *params, FILE *f)
{
	fscanf(f, "%d %d %lf %lf %d %lf\n", &(params->N), &(params->K), &(params->T), &(params->dT), &(params->LIMIT), &(params->QM));
}

/*
	Master read all points' data from the file: vector of location and speed
*/
void Master_readPoints(Point *points, int numOfPoints, FILE* f)
{
	Point *currentPoint;
	Vector *currentLocation;
	Vector *currentSpeed;
	int i;
	for (i = 0; i < numOfPoints; i++)
	{
		currentPoint = points + i;
		currentLocation = &(currentPoint->location);
		currentSpeed = &(currentPoint->speed);
		fscanf(f, "%lf %lf %lf %lf %lf %lf\n", &(currentLocation->x), &(currentLocation->y), &(currentLocation->z),
			&(currentSpeed->x), &(currentSpeed->y), &(currentSpeed->z));
	}
}

/*
	The method splits as so that master will do his code, and the slaves will do their code
*/
static void MASTER_SLAVE_Search(int id, int numproc, Params *params, Point *points, char *outputFileName)
{
	if (id == MASTER)
		MasterProccess(numproc, params, points, outputFileName);  // in "Master.h"
	else
		SlaveProccess(params, points, id);  // in "Slave.h"
}


