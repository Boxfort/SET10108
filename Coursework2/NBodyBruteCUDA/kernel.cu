#include "stdafx.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

const unsigned int N	   = 100;				// Number of bodies.
const unsigned int ITERS   = 1000;				// Number of simulation iterations.
const unsigned int THREADS = 32;				// Number of threads per block.
const unsigned int BLOCKS  = ceil(N / THREADS);	// Number of blocks required to satisfy N bodies with THREADS threads per block.

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

__global__ void n_body(unsigned int *iterations, float *pi, curandState *state)
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

void initBodies(float2* pos, float2* vel, float* mass)
{
	curandGenerator_t rnd;
	curandCreateGenerator(&rnd, CURAND_RNG_QUASI_SOBOL32);

	for (int i = 1; i < N; i++)
	{
		pos[i].x = 2.0f * (rand() / (double)RAND_MAX) - 1.0f; // Init at position between -1 : 1
		pos[i].y = 2.0f * (rand() / (double)RAND_MAX) - 1.0f; // Init at position between -1 : 1
		vel[i].x = 0;
		vel[i].y = 0;
		mass[i]  = rand() % (500 - 100 + 1) + 100;
	}

	pos[0].x = 0;
	pos[0].y = 0;
	vel[0].x = 0.0;
	vel[0].y = 0.0;
	mass[0]  = 3000;
}

int main()
{
	//Init CUDA - select device
	cudaSetDevice(0);
	cuda_info();

	// Declare host memory 
	float2 *host_pos;			// out
	float2 *host_vel;			// out
	float  *host_mass;			// out
	int	   *host_iterations;	// in

	// Allocate host memory
	cudaMallocHost((void **)&host_pos,		  (sizeof(float2) * N) * ITERS);
	cudaMallocHost((void **)&host_vel,		  (sizeof(float2) * N) * ITERS);
	cudaMallocHost((void **)&host_mass,		  (sizeof(float) * N) * ITERS);
	cudaMallocHost((void **)&host_iterations, (sizeof(unsigned int)));

	// Initialise host memory
	// Random points
	host_iterations[0] = ITERS;

	// Declare device memory
	float2 *dev_pos;			// out
	float2 *dev_vel;			// out
	float  *dev_mass;			// out
	int    *dev_iterations;		// in

	// Allocate device memory
	cudaMalloc((void**)&dev_pos,		(sizeof(float2) * N) * ITERS);
	cudaMalloc((void**)&dev_vel,		(sizeof(float2) * N) * ITERS);
	cudaMalloc((void**)&dev_mass,		(sizeof(float) * N)  * ITERS);
	cudaMalloc((void**)&dev_iterations, (sizeof(unsigned int)));

	//Copy memory from host to device
	cudaMemcpy(buffer_pi_values, &pi_values[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_pi_values, &pi_values[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_pi_values, &pi_values[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_pi_values, &pi_values[0], data_size, cudaMemcpyHostToDevice);

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
