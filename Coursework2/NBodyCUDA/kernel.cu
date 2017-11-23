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
#include <device_functions.h>


using namespace std;
using namespace std::chrono;

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

__device__ float DAMPENING = 1e-9;
__device__ float TIME_STEP = 1.0;
__device__ float G		   = 6.674e-11;

__global__ void n_body(float2* pos, float2* vel, float* mass, unsigned int* n, unsigned int* iterations)
{
	// Calculate index
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float fx = 0.0, fy = 0.0;
	int offset = 0;

	// Ensure the extra threads asigned are not run to prevent heap corruption.
	if (idx < n[0])
	{
		for (int i = 0; i < iterations[0]; i++)
		{
			int body2 = idx + ((i+offset) * n[0]);
			int body2w = idx + (i * n[0]);

			// For each body
			for (int j = 0; j < n[0]; j++)
			{
				//if (idx == j) { continue; }

				int body1 = j + ((i + offset) * n[0]);

				float dx = pos[body1].x - pos[body2].x;
				float dy = pos[body1].y - pos[body2].y;
				float distance = sqrt(dx*dx + dy*dy + DAMPENING);
				
				float force = G * (mass[body1] * (mass[body2] / distance));

				fx += force * (dx / distance);
				fy += force * (dy / distance);
			}

			vel[body2w] = vel[body2];
			pos[body2w] = pos[body2];

			vel[body2w].x += TIME_STEP * (fx / mass[i]);
			vel[body2w].y += TIME_STEP * (fy / mass[i]);

			pos[body2w].x += TIME_STEP * vel[body2w].x;
			pos[body2w].y += TIME_STEP * vel[body2w].y;

			__syncthreads();

			offset = -1;
		}
	}
}

void initBodies(float2* pos, float2* vel, float* mass, unsigned int N)
{
	/*
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
	*/
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
	ofstream file, data;

	file.open("CUDATimings.csv");
	data.open("data.csv");
	file << "BODIES,CHUNKS,TIME" << endl;

	vector<int> bodies_vec = { 1000 }; //100, 250, 500, 1000, 5000 };
	vector<int> chunks_vec = { 10 };//, 25, 50, 100, 500, 1000 };
	unsigned int timing_iterations = 1;

	for (int & bodies : bodies_vec)
	{
		for (int & chunks : chunks_vec)
		{
			file << bodies << "," << chunks << ",";

			for (int t = 0; t < timing_iterations; t++)
			{
				// Variables to change
				const unsigned int N = bodies;
				const unsigned int ITERS = 1000;							// Number of simulation iterations.
				const unsigned int THREADS = 64;							// Number of threads per block.
				const unsigned int BLOCKS = ceil(N / THREADS) * THREADS;			// Number of blocks required to satisfy N bodies with THREADS threads per block.
				const unsigned int ITER_CHUNKS = chunks;					// Number of chunks to seperate iterations into
				const unsigned int ITER_CHUNK_SIZE = ITERS / ITER_CHUNKS;   // Calculated size of iteration chunks
				

				//Init CUDA - select device
				cudaSetDevice(0);
				//cuda_info();

				// Declare host memory 
				float2		 *host_pos;			// out
				float2		 *host_vel;			// out
				float		 *host_mass;		// in
				unsigned int *host_n;			// in
				unsigned int *host_iters;		// in

				// Allocate host memory
				cudaMallocHost((void **)&host_pos, (sizeof(float2) * N) * ITER_CHUNK_SIZE);
				cudaMallocHost((void **)&host_vel, (sizeof(float2) * N) * ITER_CHUNK_SIZE);
				cudaMallocHost((void **)&host_mass, (sizeof(float)  * N));
				cudaMallocHost((void **)&host_n, (sizeof(unsigned int)));
				cudaMallocHost((void **)&host_iters, (sizeof(unsigned int)));

				// Initialise host memory
				initBodies(host_pos, host_vel, host_mass, N);
				host_n[0] = N;
				host_iters[0] = ITER_CHUNK_SIZE;

				// Declare device memory
				float2		 *dev_pos;			// out
				float2		 *dev_vel;			// out
				float		 *dev_mass;			// in
				unsigned int *dev_n;			// in
				unsigned int *dev_iters;		// in

				// Allocate device memory
				cudaMalloc((void**)&dev_pos, (sizeof(float2) * N) * ITER_CHUNK_SIZE);
				cudaMalloc((void**)&dev_vel, (sizeof(float2) * N) * ITER_CHUNK_SIZE);
				cudaMalloc((void**)&dev_mass, (sizeof(float) * N));
				cudaMalloc((void**)&dev_n, (sizeof(unsigned int)));
				cudaMalloc((void**)&dev_iters, (sizeof(unsigned int)));

				auto start = system_clock::now();

				//Copy memory from host to device
				cudaMemcpy(dev_pos, &host_pos[0], (sizeof(float2) * N) * ITER_CHUNK_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_vel, &host_vel[0], (sizeof(float2) * N)  * ITER_CHUNK_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_mass, &host_mass[0], (sizeof(float)  * N), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_n, &host_n[0], (sizeof(unsigned int)), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_iters, &host_iters[0], (sizeof(unsigned int)), cudaMemcpyHostToDevice);

				// TODO: Call Kernel once, then call with dev_memory
				for (unsigned int i = 0; i < ITER_CHUNKS; i++)
				{
					// Execute Kernel
					n_body << <BLOCKS, THREADS >> > (dev_pos, dev_vel, dev_mass, dev_n, dev_iters);

					// Wait for kernal to complete
					cudaDeviceSynchronize();

					// Read output buffer to host
					cudaMemcpy(&host_pos[0], dev_pos, (sizeof(float2) * N) * ITER_CHUNK_SIZE, cudaMemcpyDeviceToHost);
					cudaMemcpy(&host_vel[0], dev_vel, (sizeof(float2) * N) * ITER_CHUNK_SIZE, cudaMemcpyDeviceToHost);

					// Write to file
					
					for (int i = 0; i < ITER_CHUNK_SIZE; i++)
					{
						for (int k = 0; k < N; k++)
						{
							unsigned int j = k + (i * N);

							data << i << "," << host_pos[j].x << "," << host_pos[j].y << "," << host_vel[j].x << "," << host_vel[j].y << "," << host_mass[k] << endl;
						}
					}
					
				}

				auto end = system_clock::now();
				auto total = end - start;
				cout << "Bodies: " << N << " Chunks: " << ITER_CHUNKS << " Time Taken: " << duration_cast<milliseconds>(total).count() << "ms" << endl;
				file << duration_cast<milliseconds>(total).count() << ",";

				cudaFree(dev_pos);
				cudaFree(dev_vel);
				cudaFree(dev_mass);
				cudaFree(dev_n);
			}

			file << endl;
		}
	}

	file.close();

	return 0;
}
