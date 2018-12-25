
#include "Structs_MPI.h"


// VECTOR
MPI_Datatype createVectorTypeF_MPI()
{
	// Vector Struct MPI:
	const int n_vec = 3;  // for: x, y, z
	MPI_Datatype MPI_Vector;
	MPI_Datatype dataTypes[n_vec] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocksLengths[n_vec] = { 1, 1, 1 };
	MPI_Aint offsets[n_vec];

	offsets[0] = offsetof(Vector, x);
	offsets[1] = offsetof(Vector, y);
	offsets[2] = offsetof(Vector, z);

	MPI_Type_create_struct(n_vec, blocksLengths, offsets, dataTypes, &MPI_Vector);
	MPI_Type_commit(&MPI_Vector);

	return MPI_Vector;
}

// POINT
MPI_Datatype createPointType_MPI()
{
	MPI_Datatype MPI_Vector = createVectorTypeF_MPI();
	// Point Struct MPI:
	const int n_point = 3;
	MPI_Datatype MPI_Point;
	MPI_Datatype dataTypes[n_point] = { MPI_INT, MPI_Vector, MPI_Vector };
	int blocksLengths[n_point] = { 1, 1, 1 };
	MPI_Aint offsets[n_point];

	offsets[0] = offsetof(Point, clusterID);
	offsets[1] = offsetof(Point, location);
	offsets[2] = offsetof(Point, speed);

	MPI_Type_create_struct(n_point, blocksLengths, offsets, dataTypes, &MPI_Point);
	MPI_Type_commit(&MPI_Point);

	return MPI_Point;
}


// PARAMS
MPI_Datatype createParamsType_MPI()
{
	const int n_pic = 6;
	MPI_Datatype MPI_Params;
	MPI_Datatype dataTypes[n_pic] = { MPI_INT, MPI_INT, MPI_INT,
		MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocksLengths[n_pic] = { 1, 1, 1, 1, 1, 1, };
	MPI_Aint offsets[n_pic];

	offsets[0] = offsetof(Params, N);
	offsets[1] = offsetof(Params, K);
	offsets[2] = offsetof(Params, LIMIT);
	offsets[3] = offsetof(Params, QM);
	offsets[4] = offsetof(Params, T);
	offsets[5] = offsetof(Params, dT);

	MPI_Type_create_struct(n_pic, blocksLengths, offsets, dataTypes, &MPI_Params);
	MPI_Type_commit(&MPI_Params);
	return MPI_Params;
}
