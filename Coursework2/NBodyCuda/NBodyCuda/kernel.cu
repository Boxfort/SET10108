#include "stdafx.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <fstream>
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

const unsigned int N	   = 1000;				// Number of bodies.
const unsigned int ITERS   = 1000;				// Number of simulation iterations.
const unsigned int THREADS = 64;				// Number of threads per block.
const unsigned int BLOCKS  = ceil(N / THREADS) + 1;	// Number of blocks required to satisfy N bodies with THREADS threads per block.

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

/*
void calculateForces(Body* bodies)
{
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < N; i++)
	{
		if (bodies[i].dead) { continue; }

		double fx = 0.0, fy = 0.0;

		for (int j = 0; j < N; j++)
		{
			if (i == j) { continue; }
			if (bodies[j].dead) { continue; }

			double dx = bodies[j].x - bodies[i].x;
			double dy = bodies[j].y - bodies[i].y;
			double distance = sqrt(dx*dx + dy*dy + DAMPENING);

			double force = G * (bodies[j].mass * (bodies[i].mass / distance));

			fx += force * (dx / distance);
			fy += force * (dy / distance);
		}

		bodies[i].vx += TIME_STEP * (fx / bodies[i].mass);
		bodies[i].vy += TIME_STEP * (fy / bodies[i].mass);

		bodies[i].x += TIME_STEP * bodies[i].vx;
		bodies[i].y += TIME_STEP * bodies[i].vy;
	}
}

*/
__device__ float DAMPENING = 1e-9;
__device__ float TIME_STEP = 1.0;
__device__ float G		   = 6.674e-11;

__global__ void n_body(float2* pos, float2* vel, float* mass, unsigned int* n)
{
	// Calculate index
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	double fx = 0.0, fy = 0.0;

	// Ensure the extra threads asigned are not run to prevent heap corruption.
	if (idx < n[0])
	{
		// For each body
		for (int j = 0; j < n[0]; j++)
		{
			//if (idx == j) { continue; }

			double dx = pos[j].x - pos[idx].x;
			double dy = pos[j].y - pos[idx].y;
			double distance = sqrt(dx*dx + dy*dy + DAMPENING);

			double force = G * (mass[j] * (mass[idx] / distance));

			fx += force * (dx / distance);
			fy += force * (dy / distance);
		}

		vel[idx].x += TIME_STEP * (fx / mass[idx]);
		vel[idx].y += TIME_STEP * (fy / mass[idx]);

		pos[idx].x += TIME_STEP * vel[idx].x;
		pos[idx].y += TIME_STEP * vel[idx].y;
	}
}

void initBodies(float2* pos, float2* vel, float* mass)
{
	float* random_pos;
	float* random_mass;
	cudaMallocHost((void **)&random_pos,  (sizeof(float) * N) * 2);
	cudaMallocHost((void **)&random_mass, (sizeof(float) * N));

	curandGenerator_t rnd;
	curandCreateGenerator(&rnd, CURAND_RNG_PSEUDO_DEFAULT);//CURAND_RNG_QUASI_SOBOL32);
	curandSetQuasiRandomGeneratorDimensions(rnd, 1);
	curandSetGeneratorOrdering(rnd, CURAND_ORDERING_PSEUDO_DEFAULT);//CURAND_ORDERING_QUASI_DEFAULT);

	curandGenerateUniform(rnd, (float*)random_pos, N * 2);
	curandGenerateUniform(rnd, (float*)random_mass, N);

	curandDestroyGenerator(rnd);

	for (int i = 1; i < N; i++)
	{
		pos[i].x = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;//(2.0f * random_pos[i])     - 1.0f; // Init at position between -1 : 1
		pos[i].y = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;//(2.0f * random_pos[i + 1]) - 1.0f; // Init at position between -1 : 1
		vel[i].x = 0;
		vel[i].y = 0;
		mass[i]  = rand() % (500 - 100 + 1) + 100; //static_cast<int>(random_mass[i]) % (500 - 100 + 1) + 100;
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
	float2		 *host_pos;			// out
	float2		 *host_vel;			// out
	float		 *host_mass;		// in
	unsigned int *host_n;			// in

	// Allocate host memory
	cudaMallocHost((void **)&host_pos,  (sizeof(float2) * N));
	cudaMallocHost((void **)&host_vel,  (sizeof(float2) * N));
	cudaMallocHost((void **)&host_mass, (sizeof(float)  * N));
	cudaMallocHost((void **)&host_n,	(sizeof(unsigned int)));

	// Initialise host memory
	initBodies(host_pos, host_vel, host_mass);
	host_n[0] = N;

	// Declare device memory
	float2		 *dev_pos;			// out
	float2		 *dev_vel;			// out
	float		 *dev_mass;			// in
	unsigned int *dev_n;			// in

	// Allocate device memory
	cudaMalloc((void**)&dev_pos,  (sizeof(float2) * N));
	cudaMalloc((void**)&dev_vel,  (sizeof(float2) * N));
	cudaMalloc((void**)&dev_mass, (sizeof(float) * N));
	cudaMalloc((void**)&dev_n,	  (sizeof(unsigned int)));

	ofstream file;
	file.open("data.csv");

	auto start = system_clock::now();

	for (unsigned int i = 0; i < ITERS; i++)
	{
		//Copy memory from host to device
		cudaMemcpy(dev_pos, &host_pos[0], (sizeof(float2) * N), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_vel, &host_vel[0], (sizeof(float2) * N), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mass, &host_mass[0], (sizeof(float)  * N), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_n, &host_n[0], (sizeof(unsigned int)), cudaMemcpyHostToDevice);

		// Execute Kernel
		n_body <<<BLOCKS, THREADS>>> (dev_pos, dev_vel, dev_mass, dev_n);

		// Wait for kernal to complete
		cudaDeviceSynchronize();

		// Read output buffer to host
		cudaMemcpy(&host_pos[0], dev_pos, (sizeof(float2) * N), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_vel[0], dev_vel, (sizeof(float2) * N), cudaMemcpyDeviceToHost);

		/*
		for (int j = 0; j < N; j++)
		{
			file << i << "," << host_pos[j].x << "," << host_pos[j].y << "," << host_vel[j].x << "," << host_vel[j].y << "," << host_mass[j] << endl;
		}

		cout << "ITER " << i+1 << " of " << ITERS << endl;
		*/
	}

	auto end = system_clock::now();
	auto total = end - start;
	cout << "Time taken: " << duration_cast<milliseconds>(total).count() << "ms" << endl;

	cudaFree(dev_pos);
	cudaFree(dev_vel);
	cudaFree(dev_mass);
	cudaFree(dev_n);

	int a;
	cin >> a;

	return 0;
}
