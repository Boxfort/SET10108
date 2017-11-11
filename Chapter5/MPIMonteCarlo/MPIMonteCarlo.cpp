// MPISendReceive.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

const unsigned int MAX_STRING = 100;

double monte_carlo_pi(unsigned int iterations)
{
	// Create a random engine
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(millis.count());
	// Create a distribution - we want doubles between 0.0 and 1.0
	uniform_real_distribution<double> distribution(0.0, 1.0);

	// Keep track of the number of points in the circle
	unsigned int in_circle = 0;
	// Iterate
	for (unsigned int i = 0; i < iterations; ++i)
	{
		// Generate random point
		auto x = distribution(e);
		auto y = distribution(e);
		// Get length of vector defined - use pythagoras
		auto length = sqrt((x * x) + (y * y));
		// Check if in circle
		if (length <= 1.0)
		{
			++in_circle;
		}
	}

	auto pi = (4.0 * in_circle) / static_cast<double>(iterations);

	return pi;
}

int main()
{
	int num_procs, my_rank;
	double local_sum, global_sum;

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

	local_sum = monte_carlo_pi(static_cast<unsigned int>(pow(2, 24)));
	cout.precision(numeric_limits<double>::digits10);
	cout << my_rank << ":" << local_sum << endl;

	MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


	// Check if we are the main process
	if (my_rank == 0)
	{
		global_sum /= num_procs; 
		cout << "Pi=" << global_sum << endl;
	}

	MPI_Finalize();

	return 0;
}