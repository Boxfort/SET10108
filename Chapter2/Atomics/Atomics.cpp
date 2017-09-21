// Atomics.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <memory>
#include <vector>
#include <thread>
#include <iostream>
#include <mutex>
#include <atomic>

using namespace std;

mutex mut;

void increment(shared_ptr<atomic<int>> value)
{
	// Loop 1 million times incrementing value
	for (unsigned int i = 0; i < 1000000; ++i)
	{
		// Create a lock guard which automatically aqures a mutex
		lock_guard<mutex> lock(mut);
		// Increment value
		(*value)++;
	}
}

int main()
{
	// Create a shared int value
	auto value = make_shared<atomic<int>>(0);

	// Create number of threads hardware natively supports
	auto num_threads = thread::hardware_concurrency();
	vector<thread> threads;

	for (unsigned int i = 0; i < num_threads; ++i)
	{
		threads.push_back(thread(increment, value));
	}

	// Join the threads
	for (auto &t : threads)
	{
		t.join();
	}

	// Display the value
	cout << "Value = " << *value << endl;

	return 0;
}

