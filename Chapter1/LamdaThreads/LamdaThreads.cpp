#include "stdafx.h"
#include <thread>
#include <iostream>

using namespace std;

int main()
{
	// Create a thread using a lamda expression
	thread t([] {cout << "Hello from lamda thread!" << endl; });
	// Join thread
	t.join();

    return 0;
}

