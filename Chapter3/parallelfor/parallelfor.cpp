// parallelfor.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <thread>
#include <memory>
#include <omp.h>
#include <iostream>

using namespace std;

int main()
{
	auto num_threads = thread::hardware_concurrency();
	// Number of iterations
	const int n = static_cast<int>(pow(2, 30));
	// Factor value
	double factor = 0.0;
	// Calculated pi
	double pi = 0.0;

#pragma omp parallel for num_threads(num_threads) reduction(+:pi) private(factor)
	for (int k = 0; k < n; ++k)
	{
		// Determine sign of factor
		if (k % 2 == 0)
		{
			factor = 1.0;
		}
		else
		{
			factor = -1.0;
		}
		pi += factor / (2.0 * k + 1);
	}

	// Get the final value of pi
	pi *= 4.0;

	// Show more precision
	cout.precision(numeric_limits<double>::digits10);
	cout << "pi = " << pi << endl;

	int a;
	cin >> a;

    return 0;
}

