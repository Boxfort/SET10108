#include "stdafx.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

void cuda_info()
{
	// Get cuda device;
	int device;
	cudaGetDevice(&device);

	// Get device properties
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	//Display Properties
	cout << "Name: " << properties.name << endl;
	cout << "CUDA Capability: " << properties.major << endl;
	cout << "Cores: " << properties.multiProcessorCount << endl;
	cout << "Memory: " << properties.totalGlobalMem / (1024 * 1024) << "MB" << endl;
	cout << "Clock freq: " << properties.clockRate / 1000 << "MHz" << endl;
}

__global__ void monte_carlo_pi(unsigned int *iterations, float *pi, curandState *state)
{
	// Calculate index
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int start = idx * iterations[0];
	unsigned int end = start + iterations[0];

	curand_init(111, idx, 0, &state[idx]);  // Initialize CURAND

	//Set start result to 0
	unsigned int points_in_circle = 0;

	for (unsigned int i = start; i < end; ++i)
	{
		//Get point to work on
		float2 point = make_float2(curand_uniform(&state[idx]), curand_uniform(&state[idx]));
		// Calculate length
		float l = sqrtf((point.x * point.x) + (point.y * point.y));
		// Check if length and add to result accordingly
		if (l <= 1.0f)
			++points_in_circle;
	}

	pi[idx] = 4.0f * points_in_circle / (float)iterations[0]; // return estimate of pi
}

int main()
{
	const unsigned int POINTS = pow(2, 24);
	const unsigned int THREADS = 256;
	const unsigned int POINTS_PER_THREAD = POINTS / THREADS;

	//Init CUDA - select device
	cudaSetDevice(0);
	cuda_info();

	// Create host memory
	auto data_size = sizeof(float) * THREADS;
	curandState *devStates;
	vector<float> pi_values(THREADS); // Out

	// Declare buffers
	float *buffer_pi_values;
	unsigned int *buffer_points_per_thread;

	// Init Buffers
	cudaMalloc((void**)&buffer_pi_values, data_size);
	cudaMalloc((void**)&buffer_points_per_thread, sizeof(unsigned int));
	cudaMalloc((void**)&devStates, THREADS * sizeof(curandState));

	//Copy memory from host to device
	cudaMemcpy(buffer_pi_values, &pi_values[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_points_per_thread, &POINTS_PER_THREAD, sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Execute Kernel
	monte_carlo_pi << <1, THREADS >> >(buffer_points_per_thread, buffer_pi_values, devStates);

	// Wait for kernal to complete
	cudaDeviceSynchronize();

	// Read output buffer to host
	cudaMemcpy(&pi_values[0], buffer_pi_values, data_size, cudaMemcpyDeviceToHost);

	float pi_estimate = 0.0f;

	for (int i = 0; i < THREADS; i++)
	{
		pi_estimate += pi_values[i];
	}

	 pi_estimate /= THREADS;

	cout << "pi = " << pi_estimate << endl;

	cudaFree(buffer_pi_values);
	cudaFree(buffer_points_per_thread);
	cudaFree(devStates);

	int a;
	cin >> a;

	return 0;
}
