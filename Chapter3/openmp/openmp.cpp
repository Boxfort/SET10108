// openmp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <omp.h>

using namespace std;

const int NUM_THREADS = 10;

void hello()
{
	auto my_rank = omp_get_thread_num();
	auto thread_count = omp_get_num_threads();

	cout << "Helloo from thread " << my_rank << " of " << thread_count << endl;
}

int main()
{

#pragma omp parallel num_threads(NUM_THREADS)
hello();

	int a;
	cin >> a;

    return 0;
}

