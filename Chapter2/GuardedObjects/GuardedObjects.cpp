// GuardedObjects.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "guarded.h"
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace std;

const unsigned int NUM_ITERATIONS = 1000000;
const unsigned int NUM_THREADS = 4;

void task(shared_ptr<guarded> g)
{
	//Increment guarded object NUM_ITERATIONS times
	for (unsigned int i = 0; i < NUM_ITERATIONS; ++i)
	{
		g->increment();
	}
}

int main()
{
	auto g = make_shared<guarded>();

	vector<thread> threads;
	for (unsigned int i = 0; i < NUM_THREADS; ++i)
	{
		threads.push_back(thread(task, g));
	}

	for (auto &t : threads)
	{
		t.join();
	}

	cout << "Value = " << g->get_value() << endl;

	int a = 0;
	cin >> a;
    return 0;
}

