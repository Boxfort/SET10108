// parallelbubble.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;

vector<unsigned int> generate_values(unsigned int size)
{
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(static_cast<unsigned int>(millis.count()));

	vector<unsigned int> data;

	for (unsigned int i = 0; i < size; ++i)
		data.push_back(e());

	return data;
}

vector<unsigned int> bubble_sort(vector<unsigned int> list)
{

	for (unsigned int i = list.size(); i >= 2; --i)
	{
		for (unsigned int j = 0; j < (i - 1); j++)
		{
			if (list[j] > list[j + 1])
			{
				unsigned int tmp = list[j];
				list[j] = list[j + 1];
				list[j + 1] = tmp;
			}
		}
	}

	return list;
}

int main()
{
	ofstream results("bubble.csv", ofstream::out);

	for (unsigned int size = 8; size <= 16; ++size)
	{
		results << pow(2, size) << ", ";

		for (unsigned int i = 0; i < 100; ++i)
		{
			cout << "Generating " << i << " for " << pow(2, size) << " values" << endl;
			auto data = generate_values(static_cast<unsigned int>(pow(2, size)));

			cout << "Sorting" << endl;

			auto start = system_clock::now();
			bubble_sort(data);
			auto end = system_clock::now();
			auto total = duration_cast<milliseconds>(end - start).count();

			results << total << ",";
		}
		results << endl;
	}
	results.close();
    return 0;
}

