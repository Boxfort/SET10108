// ThreadSafeDataStructures.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "threadsafe_stack.h"
#include <thread>
#include <iostream>
#include <memory>

using namespace std;

void pusher(shared_ptr<threadsafe_stack<unsigned int>> stack)
{
	for (unsigned int i = 0; i < 1000000; ++i)
	{
		stack->push(i);
	}
}

void popper(shared_ptr<threadsafe_stack<unsigned int>> stack)
{
	unsigned int count = 0;
	while (count < 1000000)
	{
		try
		{
			auto val = stack->pop();
			++count;
		}
		catch (exception e)
		{
			cout << e.what() << endl;
		}
	}
}

int main()
{
	auto stack = make_shared<threadsafe_stack<unsigned int>>();
	thread t1(popper, stack);
	thread t2(pusher, stack);

	t1.join();
	t2.join();

	cout << "Stack empty = " << stack->empty() << endl;

	int a=0;
	cin >> a;

    return 0;
}

