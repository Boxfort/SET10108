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
}

void initBodies(float2* pos, float2* vel, float* mass)
{
	float* random_pos;
	float* random_mass;
	cudaMallocHost((void **)&random_pos,  (sizeof(float) * N) * 2);
	cudaMallocHost((void **)&random_mass, (sizeof(float) * N));

	curandGenerator_t rnd;
	curandCreateGenerator(&rnd, CURAND_RNG_QUASI_SOBOL32);
	curandSetQuasiRandomGeneratorDimensions(rnd, 1);
	curandSetGeneratorOrdering(rnd, CURAND_ORDERING_QUASI_DEFAULT);

	curandGenerateUniform(rnd, (float*)random_pos, N * 2);
	curandGenerateUniform(rnd, (float*)random_mass, N);

	for (int i = 1; i < N; i++)
	{
		pos[i].x = (2.0f * random_pos[i]) - 1.0f; // Init at position between -1 : 1
		pos[i].y = (2.0f * random_pos[i + 1]) - 1.0f; // Init at position between -1 : 1
		vel[i].x = 0;
		vel[i].y = 0;
		mass[i]  = static_cast<int>(random_mass[i]) % (500 - 100 + 1) + 100;
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
	float  *host_mass;			// in
	int	   *host_iterations;	// in

	// Allocate host memory
	cudaMallocHost((void **)&host_pos,		  (sizeof(float2) * N) * ITERS);
	cudaMallocHost((void **)&host_vel,		  (sizeof(float2) * N) * ITERS);
	cudaMallocHost((void **)&host_mass,		  (sizeof(float) * N) * ITERS);
	cudaMallocHost((void **)&host_iterations, (sizeof(unsigned int)));

	// Initialise host memory
	initBodies(host_pos, host_vel, host_mass);
	host_iterations[0] = ITERS;

	// Declare device memory
	float2		 *dev_pos;			// out
	float2		 *dev_vel;			// out
	float		 *dev_mass;			// in
	unsigned int *dev_iterations;	// in

	// Allocate device memory
	cudaMalloc((void**)&dev_pos,		(sizeof(float2) * N) * ITERS);
	cudaMalloc((void**)&dev_vel,		(sizeof(float2) * N) * ITERS);
	cudaMalloc((void**)&dev_mass,		(sizeof(float) * N)  * ITERS);
	cudaMalloc((void**)&dev_iterations, (sizeof(unsigned int)));

	//Copy memory from host to device
	cudaMemcpy(dev_pos,		   &host_pos[0],		(sizeof(float2) * N) * ITERS, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vel,		   &host_vel[0],		(sizeof(float2) * N) * ITERS, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mass,	   &host_mass[0],		(sizeof(float) * N)  * ITERS, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_iterations, &host_iterations[0], (sizeof(unsigned int)),		  cudaMemcpyHostToDevice);

	// Execute Kernel
	n_body << <BLOCKS, THREADS >> >(dev_pos, dev_vel, dev_mass, dev_iterations);

	// Wait for kernal to complete
	cudaDeviceSynchronize();

	// Read output buffer to host
	cudaMemcpy(&host_pos[0], dev_pos, (sizeof(float2) * N) * ITERS, cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_vel[0], dev_vel, (sizeof(float2) * N) * ITERS, cudaMemcpyDeviceToHost);

	cudaFree(dev_pos);
	cudaFree(dev_vel);
	cudaFree(dev_mass);
	cudaFree(dev_iterations);

	int a;
	cin >> a;

	return 0;
}
