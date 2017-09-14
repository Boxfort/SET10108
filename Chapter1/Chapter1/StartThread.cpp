#include "stdafx.h"
#include <iostream>
#include <thread>

using namespace std;

/*
This is the method called by the thread.
*/
void hello_world()
{
	cout << "Hello World! Thread : " << this_thread::get_id() << endl;
}

int main()
{
	// Create a new thread
	thread t(hello_world);
	// Wait for thread to finish (join it)
	t.join();
	// Return OK (0)
    return 0;
}

