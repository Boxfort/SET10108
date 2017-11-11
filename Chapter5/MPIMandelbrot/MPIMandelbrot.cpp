// MPIMandelbrot.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <vector>

using namespace std;

// Number of iterations to find each pixel value.
const unsigned int MAX_ITERATIONS = 1000;
// Dimension of the image to generate (in pixels)
const unsigned int DIMENSION = 1024;
const unsigned int SPLIT_AMOUNT = 4;
const unsigned int SUB_DIMENSION = DIMENSION / SPLIT_AMOUNT;


const int BPP = 24;

// Mandelbrot dimesnions are ([-2.1, 1.0], [-1.3, 1.3])
const double X_MIN = -2.1;
const double X_MAX = 1.0;
const double Y_MIN = -1.3;
const double Y_MAX = 1.3;

// Convert from mandlebrot coordinate to image coordinate
const double INTEGRAL_X = (X_MAX - X_MIN) / static_cast<double>(DIMENSION);
const double INTEGRAL_Y = (Y_MAX - Y_MIN) / static_cast<double>(DIMENSION);

double* mandelbrot(unsigned int start_y, unsigned int end_y)
{
	int arr_size = (end_y - start_y) * DIMENSION;

	// Declare the values that we will use
	double x, y, x1, y1, xx = 0.0;
	unsigned int loop_count = 0;
	unsigned int arr_pos = 0;

	double* results = new double[arr_size];

	// Loop through each line

	y = Y_MIN + (start_y * INTEGRAL_Y);
	for (unsigned int y_coord = start_y; y_coord < end_y; ++y_coord)
	{
		x = X_MIN;
		// Loop through each pixel on the line.
		for (unsigned int x_coord = 0; x_coord < DIMENSION; ++x_coord)
		{
			x1 = 0.0, y1 = 0.0;
			loop_count = 0;

			while (loop_count < MAX_ITERATIONS && sqrt((x1 * x1) + (y1 * y1)) < 2.0)
			{
				++loop_count;
				xx = (x1 * x1) - (y1 * y1) + x;
				y1 = 2 * x1 * y1 + y;
				x1 = xx;
			}
			// Get Value after loop has completed
			double val = static_cast<double>(loop_count) / static_cast<double>(MAX_ITERATIONS);

			// Push value to results
			results[arr_pos] = val;
			arr_pos++;
			// Increase x based on integral
			x += INTEGRAL_X;
		}

		y += INTEGRAL_Y;
	}

	return results;
}


int main()
{
	int num_procs, my_rank;

	// Init MPI
	auto result = MPI_Init(nullptr, nullptr);
	if (result != MPI_SUCCESS)
	{
		cout << "ERROR - initialising MPI" << endl;
		MPI_Abort(MPI_COMM_WORLD, result);
		return -1;
	}

	// Get MPI info
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//Split dimension by numprocs
	//for each prop
		//send starty = chunk size *num endy = chunksize + 1 * num

	int chunk_size = DIMENSION / num_procs;

	double* data = new double[DIMENSION * DIMENSION];
	double* mbresult = mandelbrot(chunk_size * my_rank, chunk_size * (my_rank + 1));

	MPI_Gather(&mbresult[0], DIMENSION / num_procs, MPI_DOUBLE,// Source
		&data[0], DIMENSION / num_procs, MPI_DOUBLE,// Dest
		0, MPI_COMM_WORLD);

	delete &mbresult;
	delete &data;

	// Check if we are the main process
	if (my_rank == 0)
	{
		// Not main process

		cout << "Got em" << endl;
	}

	MPI_Finalize();

	return 0;
}