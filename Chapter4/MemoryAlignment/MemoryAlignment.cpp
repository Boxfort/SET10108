// MemoryAlignment.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <xmmintrin.h> 
#include <iostream>

using namespace std;

int main()
{
	// Declare a single 128-bit value aligned to 16 bytes ( size of 128-bits)
	__declspec(align(16)) __m128 x;
	// We can treat x as a collection of  four floats,
	// or other combinations of values for 128-bits
	x.m128_f32[0] = 10.0f;
	x.m128_f32[1] = 20.0f;
	x.m128_f32[2] = 30.0f;
	x.m128_f32[3] = 40.0f;

	// We can print out individual numbers
	cout << x.m128_f32[0] << endl;

	// The key is that the memory is aligned - it is faster to access in blocks.
	// Create an array of SIZE floats aligned to 4 bytes (size of a float)
	const int SIZE = 4;
	float* data = (float*)_aligned_malloc(SIZE * sizeof(float), 4);

	// Access it just like an array
	cout << data[0] << endl;

	//Create an array of SIZE 128-bit values aligned to 16 bytes
	__m128* big_data = (__m128*)_aligned_malloc(SIZE * sizeof(__m128), 16);

	//Access just like an araayt of __m128
	cout << big_data[0].m128_f32[0] << endl;

	// Free the data - ALWAYS REMEMBER TO FREE YOUR MEMORY!!!!
	// We are dealing a C level.
	_aligned_free(data);
	_aligned_free(big_data);

    return 0;
}

