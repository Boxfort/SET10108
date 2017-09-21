// ConditionVariables.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <condition_variable>
#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;

mutex mut;

void task_1(condition_variable &condition)
{
	// Task one initially sleeps
	cout << "Task 1 sleeping for 3 secs" << endl;
	this_thread::sleep_for(seconds(3));
	// Notify waiting thread
	cout << "Task 1 notifying waiting thread" << endl;
	condition.notify_one();
	// Now wait for notifications
	cout << "Task 1 waiting for a notification" << endl;
	condition.wait(unique_lock<mutex>(mut));
	// We can now continue
	cout << "Task 1 notified" << endl;
	// Sleep for 3 secs
	cout << "Task 1 sleeping for 3 secs" << endl;
	this_thread::sleep_for(seconds(3));
	// Notify any waiting thread
	cout << "Task 1 notifying waiting thread" << endl;
	condition.notify_one();
	// Now wait 3 seconds for notification
	cout << "Task 1 waiting for 3 secs for notification" << endl;
	if (static_cast<int>(condition.wait_for(unique_lock<mutex>(mut), seconds(3))))
	{
		cout << "Task 1 sleeping for 3 secs" << endl;
	}
	else
	{
		cout << "Task 1 got tired waiting" << endl;
	}

	cout << "Task 1 finished" << endl;
}

void task_2(condition_variable &condition)
{
	// Task two wil initially wait for notification
	cout << "Task 2 waiting for notification" << endl;
	condition.wait(unique_lock<mutex>(mut));
	// We are free to continue
	cout << "Task 2 notified" << endl;
	//Sleep for 5 seconds
	cout << "Task 2 sleeping for 5 secs" << endl;
	this_thread::sleep_for(seconds(5));
	// Notifying waiting thread
	cout << "Task 2 notifying waiting thread" << endl;
	condition.notify_one();
	// Now wait 5 seconds for notification
	cout << "Task 2 waiting 5 secs for notifications" << endl;
	if (static_cast<int>(condition.wait_for(unique_lock<mutex>(mut), seconds(5)))) 
	{
		cout << "Task 2 notified before 5 seconds" << endl;
	}
	else
	{
		cout << "Task 2 got tired of waiting" << endl;
	}

	// Sleep for 5 seconds
	cout << "Task 2 sleeping for 5 secs" << endl;
	this_thread::sleep_for(seconds(5));
	// Notify waiting thread
	cout << "Task 2 notifying waiting thread" << endl;
	condition.notify_one();
	// Print finished
	cout << "Task 2 Finished" << endl;
}

int main()
{
	//Create condition variable
	condition_variable condition;

	//create two threads
	thread t1(task_1, ref(condition));
	thread t2(task_2, ref(condition));

	//join threads
	t1.join();
	t2.join();

    return 0;
}

