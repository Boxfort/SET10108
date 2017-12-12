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

__global__ void n_body(float* pos_x, float* vel_x, float* pos_y, float* vel_y, float* mass, unsigned int* n, unsigned int* iterations)
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
			// Previous iterations body.
			int body2 = idx + ((i+offset) * n[0]);
			// Current iterations body
			int body2w = idx + (i * n[0]);

			for (int j = 0; j < n[0]; j++)
			{
				// Previous iteration body
				int body1 = j + ((i + offset) * n[0]);

				// Calculate distances between bodies
				float dx = pos_x[body1] - pos_x[body2];
				float dy = pos_y[body1] - pos_y[body2];
				float distance = sqrt(dx*dx + dy*dy + DAMPENING);
				
				// Calulate forces on bodies based on mass and distance

				float force = G * (mass[idx] * (mass[j] / distance));

				fx += force * (dx / distance);
				fy += force * (dy / distance);
			}

			// Copy previous iteration state to this iteration
			vel_x[body2w] = vel_x[body2];
			pos_x[body2w] = pos_x[body2];
			vel_y[body2w] = vel_y[body2];
			pos_y[body2w] = pos_y[body2];

			// Move bodies based on force acting upon them.
			vel_x[body2w] += TIME_STEP * (fx / mass[idx]);
			vel_y[body2w] += TIME_STEP * (fy / mass[idx]);

			pos_x[body2w] += TIME_STEP * vel_x[body2w];
			pos_y[body2w] += TIME_STEP * vel_y[body2w];

			__syncthreads();

			//After the first iteration start referencing previous iterations state
			offset = -1;
		}
	}
}

void initBodies(float* pos_x, float* pos_y, float* vel_x, float* vel_y, float* mass, unsigned int N)
{
	// Initialise every body with a random location and mass and no velocity.

	for (int i = 0; i < N; i++)
	{
		pos_x[i] = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;//(2.0f * random_pos[i])     - 1.0f; // Init at position between -1 : 1
		pos_y[i] = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;//(2.0f * random_pos[i + 1]) - 1.0f; // Init at position between -1 : 1
		vel_x[i] = 0;
		vel_y[i] = 0;
		mass[i] = rand() % (500 - 100 + 1) + 100; //static_cast<int>(random_mass[i]) % (500 - 100 + 1) + 100;
	}
	// Create a large body in the center.
	pos_x[0] = 0;
	pos_y[0] = 0;
	mass[0] = 4000;
}

int main()
{
	ofstream file, data;

	file.open("CUDATimingsChunks.csv");
	data.open("data.csv");
	file << "THREADS,CHUNKS,TIME" << endl;

	vector<int> threads_vec = { 32 }; //, 64 };
	vector<int> chunks_vec = { 2 };//, 5, 10, 25, 50, 100, 500, 1000 };
	unsigned int timing_iterations = 1;

	for (int & threads : threads_vec)
	{
		for (int & chunks : chunks_vec)
		{
			file << threads << "," << chunks << ",";

			for (int t = 0; t < timing_iterations; t++)
			{
				// Variables to change
				const unsigned int N = 128;
				const unsigned int ITERS = 1000;							// Number of simulation iterations.
				const unsigned int THREADS = threads;						// Number of threads per block.
				const unsigned int BLOCKS = ceil(N / THREADS) * THREADS;	// Number of blocks required to satisfy N bodies with THREADS threads per block.
				const unsigned int ITER_CHUNKS = chunks;					// Number of chunks to seperate iterations into
				const unsigned int ITER_CHUNK_SIZE = ITERS / ITER_CHUNKS;   // Calculated size of iteration chunks


				//Init CUDA - select device
				cudaSetDevice(0);
				cuda_info();

				// Declare host memory 
				float		 *host_pos_x;		// out
				float		 *host_vel_x;		// out
				float		 *host_pos_y;		// out
				float		 *host_vel_y;		// out
				float		 *host_mass;		// in
				unsigned int *host_n;			// in
				unsigned int *host_iters;		// in

				// Allocate host memory
				cudaMallocHost((void **)&host_pos_x, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMallocHost((void **)&host_vel_x, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMallocHost((void **)&host_pos_y, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMallocHost((void **)&host_vel_y, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMallocHost((void **)&host_mass, (sizeof(float)  * N));
				cudaMallocHost((void **)&host_n, (sizeof(unsigned int)));
				cudaMallocHost((void **)&host_iters, (sizeof(unsigned int)));

				// Initialise host memory
				initBodies(host_pos_x, host_pos_y, host_vel_x, host_vel_y, host_mass, N);
				host_n[0] = N;
				host_iters[0] = ITER_CHUNK_SIZE;

				// Declare device memory
				float		 *dev_pos_x;		// out
				float		 *dev_vel_x;		// out
				float		 *dev_pos_y;		// out
				float		 *dev_vel_y;		// out
				float		 *dev_mass;			// in
				unsigned int *dev_n;			// in
				unsigned int *dev_iters;		// in

				// Allocate device memory
				cudaMalloc((void**)&dev_pos_x, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMalloc((void**)&dev_vel_x, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMalloc((void**)&dev_pos_y, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMalloc((void**)&dev_vel_y, (sizeof(float) * N) * ITER_CHUNK_SIZE);
				cudaMalloc((void**)&dev_mass, (sizeof(float) * N));
				cudaMalloc((void**)&dev_n, (sizeof(unsigned int)));
				cudaMalloc((void**)&dev_iters, (sizeof(unsigned int)));

				auto start = system_clock::now();

				//Copy memory from host to device
				cudaMemcpy(dev_pos_x, &host_pos_x[0], (sizeof(float) * N) * ITER_CHUNK_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_vel_x, &host_vel_x[0], (sizeof(float) * N)  * ITER_CHUNK_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_pos_y, &host_pos_y[0], (sizeof(float) * N) * ITER_CHUNK_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_vel_y, &host_vel_y[0], (sizeof(float) * N)  * ITER_CHUNK_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_mass, &host_mass[0], (sizeof(float)  * N), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_n, &host_n[0], (sizeof(unsigned int)), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_iters, &host_iters[0], (sizeof(unsigned int)), cudaMemcpyHostToDevice);

				for (unsigned int i = 0; i < ITER_CHUNKS; i++)
				{
					// Execute Kernel
					n_body <<<BLOCKS, THREADS >>>(dev_pos_x, dev_vel_x, dev_pos_y, dev_vel_y, dev_mass, dev_n, dev_iters);

					// Wait for kernal to complete
					cudaDeviceSynchronize();

					// Read output buffer to host
					cudaMemcpy(&host_pos_x[0], dev_pos_x, (sizeof(float) * N) * ITER_CHUNK_SIZE, cudaMemcpyDeviceToHost);
					cudaMemcpy(&host_vel_x[0], dev_vel_x, (sizeof(float) * N) * ITER_CHUNK_SIZE, cudaMemcpyDeviceToHost);
					cudaMemcpy(&host_pos_y[0], dev_pos_y, (sizeof(float) * N) * ITER_CHUNK_SIZE, cudaMemcpyDeviceToHost);
					cudaMemcpy(&host_vel_y[0], dev_vel_y, (sizeof(float) * N) * ITER_CHUNK_SIZE, cudaMemcpyDeviceToHost);

					auto err = cudaGetLastError();
					if(err != 0)
						cout << cudaGetErrorName(err) << endl;

					
					// Write to file
					for (int i = 0; i < ITER_CHUNK_SIZE; i++)
					{
						for (int k = 0; k < N; k++)
						{
							unsigned int j = k + (i * N);

							data << i << "," << host_pos_x[j] << "," << host_pos_y[j] << "," << host_vel_x[j] << "," << host_vel_y[j] << "," << host_mass[k] << endl;
						}
					}
					

					// Copy final iteration to first iteration
					for (int i = 0; i < N; i++)
					{
						host_pos_x[i] = host_pos_x[i + ((ITER_CHUNK_SIZE - 1) * N)];
						host_vel_x[i] = host_vel_x[i + ((ITER_CHUNK_SIZE - 1) * N)];
						host_pos_y[i] = host_pos_y[i + ((ITER_CHUNK_SIZE - 1) * N)];
						host_vel_y[i] = host_vel_y[i + ((ITER_CHUNK_SIZE - 1) * N)];

						for (int j = 0; j < ITER_CHUNK_SIZE-1; j++)
						{
							host_vel_x[i + ITER_CHUNK_SIZE + j] = 0;
							host_vel_y[i + ITER_CHUNK_SIZE + j] = 0;
						}
					}


					//Copy memory from host to device
					cudaMemcpy(dev_pos_x, &host_pos_x[0], (sizeof(float) * N), cudaMemcpyHostToDevice);
					cudaMemcpy(dev_vel_x, &host_vel_x[0], (sizeof(float) * N), cudaMemcpyHostToDevice);
					cudaMemcpy(dev_pos_y, &host_pos_y[0], (sizeof(float) * N), cudaMemcpyHostToDevice);
					cudaMemcpy(dev_vel_y, &host_vel_y[0], (sizeof(float) * N), cudaMemcpyHostToDevice);
				}

				auto end = system_clock::now();
				auto total = end - start;
				cout << "Threads: " << THREADS << " Chunks: " << ITER_CHUNKS << " Time Taken: " << duration_cast<milliseconds>(total).count() << "ms" << endl;
				file << duration_cast<milliseconds>(total).count() << ",";

				// Free allocated memory.
				cudaFree(dev_pos_x);
				cudaFree(dev_vel_x);
				cudaFree(dev_pos_y);
				cudaFree(dev_vel_y);
				cudaFree(dev_mass);
				cudaFree(dev_n);
				cudaFree(dev_iters);
			}

			file << endl;
		}
	}

	file.close();

	return 0;
}
