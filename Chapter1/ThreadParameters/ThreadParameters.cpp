#include "stdafx.h"
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int NUM_THREADS = 100;

void task(int n, int val)
{
	cout << "Thread: " << n << " Random Value: " << val << endl;
}

int main()
{
	// C++11 style of creating a random with the current time in milliseconds
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(static_cast<unsigned int>(millis.count()));

	// Create 100 threads in a vector
	vector<thread> threads;
	for (int i = 0; i < NUM_THREADS; i++)
	{
		threads.push_back(thread(task, i, e()));

	}

	// Use C++11 ranged for loop to join the threads
	// Same as foreach in C#
	for (auto &t : threads)
	{
		t.join();
	}

    return 0;
}

