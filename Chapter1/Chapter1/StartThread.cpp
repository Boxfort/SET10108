#include "stdafx.h"
#include <iostream>
#include <thread>

void hello_world()
{
	std::cout << "Hello World!" << std::endl;
}

int main()
{
	std::thread t(hello_world);
    return 0;
}

