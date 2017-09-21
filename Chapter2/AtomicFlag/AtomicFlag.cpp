// AtomicFlag.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <memory>
#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

void task(unsigned int id, shared_ptr<atomic_flag> flag)
{
	for (unsigned int i = 0; i < 10; ++i)
	{
		// Test the flag is available, and grab when it is
		// Notice this while loop keeps spinning until the flag is clear
		while (flag->test_and_set());
		// Flag is available. Thread dispalys message
		cout << "Thread " << id << " running " << i << endl;
		
		this_thread::sleep_for(seconds(1));
		//Clear flag
		flag->clear();
	}
}


int main()
{
	auto flag = make_shared<atomic_flag>();

	auto num_threads = thread::hardware_concurrency();

	vector<thread> threads;
	for (unsigned int i = 0; i < num_threads; ++i)
	{
		threads.push_back(thread(task, i, flag));
	}

	for (auto &t : threads)
	{
		t.join();
	}

    return 0;
}

