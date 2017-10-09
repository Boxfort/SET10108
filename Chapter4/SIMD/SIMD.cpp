// SIMD.cpp : Defines the entry point for the console application.
//

// -- HORIZONTAL ADD, FOR SUMMING 4 LENGTH __m128f -- 
//// xx = { xx3, xx2, xx1, xx0 }
//xx = _mm_hadd_ps(xx, xx);
//// xx = {xx3+xx2, xx1+xx0, xx3+xx2, xx1+xx0}
//xx = _mm_hadd_ps(xx, xx);
//// xx = {xx2+xx3+xx1+xx0, xx3+xx2+xx1+xx0, xx3+xx2+xx1+xx0, xx3+xx2+xx1+xx0}


#include "stdafx.h"
#include <math.h>
#include <xmmintrin.h> 
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

// Size of data to allocate - divide by four to get number of vectors
const unsigned int SIZE = static_cast<unsigned int>(pow(2, 24));
const unsigned int NUM_VECTORS = SIZE / 4;

int main()
{
	// Data - aligned to 16 bytes (128-bits)
	auto data = (float*)_aligned_malloc(SIZE * sizeof(float), 16);
	// Initilize data
	for (unsigned int i = 0; i < SIZE; ++i)
	{
		//Set all values to 1
		data[i] = 1.0f;
	}

	// Value to add to all values
	auto value = _mm_set1_ps(4.0f);

	// Pointer to the data
	auto stream_data = (__m128*)data;

	// Start timer 
	auto start = high_resolution_clock::now();

	// Add value to stream data
	for (unsigned int i = 0; i < NUM_VECTORS; ++i)
	{
		stream_data[i] = _mm_add_ps(stream_data[i], value);
	}

	auto end = high_resolution_clock::now();

	auto total = duration_cast<microseconds>(end - start).count();
	cout << "SIMD: " << total << "micros" << endl;

	//Free memory
	_aligned_free(data);

	data = new float[SIZE];
	//Set all values to 1 
	for (unsigned int i = 0; i < SIZE; ++i)
	{
		data[i] = data[i] + 4.0f;
	}

	//Start timer

	start = high_resolution_clock::now();

	for (unsigned int i = 0; i < SIZE; ++i)
	{
		data[i] = data[i] + 4.0f;
	}

	end = high_resolution_clock::now();
	total = duration_cast<microseconds>(end - start).count();
	cout << "NON-SIMD: " << total << "micros" << endl;

	delete[] data;

	int a;
	cin >> a;

    return 0;
}

