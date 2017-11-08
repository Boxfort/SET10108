#include "stdafx.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

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
	cout << "Memory: " << properties.totalGlobalMem / (1024*1024) << "MB" << endl;
	cout << "Clock freq: " << properties.clockRate / 1000 << "MHz" << endl;
}

__global__ void vecadd(const int *A, const int *B, int *C)
{
	// Get block index
	unsigned int block_idx = blockIdx.x;
	// Get thread index 
	unsigned int thread_idx = threadIdx.x;
	// Get the number of threads per block
	unsigned int block_dim = blockDim.x;
	// Get the threads unique ID - (block_idx * block_dim) + thread_idx;
	unsigned int idx = (block_idx * block_dim) + thread_idx;
	// Add corresponding locations of A and B and sotre in C
	C[idx] = A[idx] + B[idx];
}

int main()
{
	const unsigned int ELEMENTS = 2048;

	//Init CUDA - select device
	cudaSetDevice(0);
	cuda_info();

	// Create host memory
	auto data_size = sizeof(int) * ELEMENTS;
	vector<int> A(ELEMENTS); //In
	vector<int> B(ELEMENTS); //In
	vector<int> C(ELEMENTS); //Out

	// init input data
	for (unsigned int i = 0; i < ELEMENTS; ++i)
	{
		A[i] = B[i] = i;
	}

	// Declare buffers
	int *buffer_A, *buffer_B, *buffer_C;

	// Init Buffers
	cudaMalloc((void**)&buffer_A, data_size);
	cudaMalloc((void**)&buffer_B, data_size);
	cudaMalloc((void**)&buffer_C, data_size);

	//Copy memory from host to device
	cudaMemcpy(buffer_A, &A[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_B, &B[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_C, &C[0], data_size, cudaMemcpyHostToDevice);

	// Execute Kernel
	vecadd<<<ELEMENTS / 1024, 1024 >>>(buffer_A, buffer_B, buffer_C);

	// Wait for kernal to complete
	cudaDeviceSynchronize();

	// Read output buffer to host
	cudaMemcpy(&C[0], buffer_C, data_size, cudaMemcpyDeviceToHost);

	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);

	int a;
	cin >> a;

	return 0;
}
