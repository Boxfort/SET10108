#include "stdafx.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
using namespace std::chrono;

void work()
{
	// Do some spinning - no actual processing but will make the CPU work
	int n = 0;
	for (int i = 0; i < 1000000; ++i)
	{
		++n;
	}
}

int main()
{
	// Create a new file
	ofstream data("data.csv", ofstream::out);
	// We're going to gather 100 readings so create a thread and join it 100 times
	for (int i = 0; i < 100; i++)
	{
		// Get start time
		auto start = system_clock::now();
		// Execute thread
		thread t(work);
		t.join();
		// Get end time
		auto end = system_clock::now();
		// Calculate the duration
		auto total = end - start;
		// Convert time to ms and write to file
		data << duration_cast<milliseconds>(total).count() << endl;
	}

	// 100 iterations complete. CLose file
	data.close();
    return 0;
}

