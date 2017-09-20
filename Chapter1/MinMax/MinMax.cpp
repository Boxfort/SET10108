// MinMax.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <random>
#include <vector>
#include <chrono>
#include <climits>
#include <future>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;

const int NUMS_TO_GENERATE = 1000000;

vector<unsigned int> numbers;

void generate_random_numbers() 
{
	// Clear the numbers vector.
	numbers.clear();

	// Create a random engine
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(millis.count());
	// Create a distribution - we want doubles between 0 and UINT_MAX
	uniform_int_distribution<unsigned int> distribution(0, UINT_MAX);

	for (unsigned int i = 0; i < NUMS_TO_GENERATE; i++)
	{
		numbers.push_back(distribution(e));
	}
}

void get_max(promise<unsigned int> && p, unsigned int lower, unsigned int upper)
{
	unsigned int maxnum = 0;



	// find max
	for (unsigned int i = lower; i < upper; ++i)
	{

		if (numbers[i] > maxnum)
		{
			maxnum = numbers[i];
		}
	}

	//cout << maxnum << endl;
	//set max
	p.set_value(maxnum);
}

int main()
{
	// Create data file
	ofstream data("max.csv", ofstream::out);

	for (unsigned int num_threads = 0; num_threads <= 6; ++num_threads)
	{
		unsigned int max = 0;

		generate_random_numbers();

		auto total_threads = static_cast<unsigned int>(pow(2.0, num_threads));

		data << "num_threads_" << total_threads;

		// Now execute 100 iterations
		for (unsigned int iters = 0; iters < 100; ++iters)
		{
			//Divide number to chunks
			unsigned int chunk_size = NUMS_TO_GENERATE / total_threads;
			
			// Get the start time
			auto start = system_clock::now();
			// We need to create total_threads threads
			vector<thread> threads;
			vector<future<unsigned int>> futures;
			for (unsigned int n = 0; n < total_threads; n++)
			{
				cout << "Creating thread " << n << endl;

				unsigned int lower = chunk_size * n;
				unsigned int upper = chunk_size * (n + 1);

				promise<unsigned int> p;
				auto f = p.get_future();
				thread t(get_max, move(p), lower, upper);
				threads.push_back(move(t));
				futures.push_back(move(f));

			}

			cout << "Iteration: " << iters << endl;

			// Join the threads
			for (auto &t : threads)
			{
				t.join();
			}

			// Get max from futures
			for (auto &f : futures)
			{
				unsigned int val = f.get();
				bool cmon = val > max;
				if (cmon)
				{
					max = val;
				}
			}

			// Get the end time
			auto end = system_clock::now();
			// Get the total time
			auto total = end - start;
			// Convert to milliseconds and output to file
			data << ", " << duration_cast<milliseconds>(total).count();
		}
		data << ", " << max;
		data << endl;
	}

	// Close the file
	data.close();
    return 0;
}

