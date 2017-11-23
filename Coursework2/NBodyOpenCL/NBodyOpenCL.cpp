// OpenCLIntro.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <chrono>

using namespace std;
using namespace std::chrono;

void initialise_opencl(vector<cl_platform_id> &platforms, vector<cl_device_id> &devices, cl_context &context, cl_command_queue &cmd_queue)
{
	// Status of OpenCL calls
	cl_int status;

	// get the number of platforms
	cl_uint num_platforms;
	status = clGetPlatformIDs(0, nullptr, &num_platforms);
	// Resize vector to store platforms
	platforms.resize(num_platforms);
	// Fill in platform vector
	status = clGetPlatformIDs(num_platforms, &platforms[0], nullptr);

	// Assume platform 0 is the one we want to use
	// Get devies for platform 0
	cl_uint num_devices;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
	// Resize vector to store devices
	devices.resize(num_devices);
	// Fill in devices vector
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, &devices[0], nullptr);

	// Create a context
	context = clCreateContext(nullptr, num_devices, &devices[0], nullptr, nullptr, &status);

	// Create a command queue
	cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);
}

void print_opencl_info(vector<cl_device_id> &devices)
{
	// Buffers for device name and vendor
	char device_name[1024], vendor[1024];
	// Declare other necessary variables
	cl_uint num_cores;
	cl_long memory;
	cl_uint clock_freq;
	cl_bool available;

	// Iterate through eac hdevice in vector and display
	for (auto &d : devices)
	{
		// Get info
		clGetDeviceInfo(d, CL_DEVICE_NAME, 1024, device_name, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_VENDOR, 1024, vendor, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cores, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_long), &memory, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock_freq, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, nullptr);
	}

	// Print info
	cout << "Device: " << device_name << endl;
	cout << "Vendor: " << vendor << endl;
	cout << "Cores: " << num_cores << endl;
	cout << "Memory: " << memory << endl;
	cout << "Clock freq: " << clock_freq << endl;
	cout << "Available: " << available << endl;
	cout << "**************************" << endl << endl;
}

cl_program load_program(const string &filename, cl_context &context, cl_device_id &device, cl_int num_devices)
{
	// Status of CL calls
	cl_int status;

	// Create and compile program
	// Read in kernel
	ifstream input(filename, ifstream::in);
	stringstream buffer;
	buffer << input.rdbuf();
	// Get the char array of the file contents
	auto file_contents = buffer.str();
	auto char_contents = file_contents.c_str();

	// Create program object
	auto program = clCreateProgramWithSource(context, 1, &char_contents, nullptr, &status);

	// Compile program
	status = clBuildProgram(program, num_devices, &device, nullptr, nullptr, nullptr);

	// Check if compiled
	if (status != CL_SUCCESS)
	{
		// Error
		size_t length;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
		char *log = new char[length];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log, &length);
		cout << log << endl;
		delete[] log;
	}

	return program;
}

void initBodies(float* pos_x, float* pos_y, float* vel_x, float* vel_y , float* mass, unsigned int N)
{
	for (int i = 1; i < N; i++)
	{
		pos_x[i] = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;//(2.0f * random_pos[i])     - 1.0f; // Init at position between -1 : 1
		pos_y[i] = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;//(2.0f * random_pos[i + 1]) - 1.0f; // Init at position between -1 : 1
		vel_x[i] = 0;
		vel_y[i] = 0;
		mass[i] = rand() % (500 - 100 + 1) + 100; //static_cast<int>(random_mass[i]) % (500 - 100 + 1) + 100;
	}

	pos_x[0] = 0;
	pos_y[0] = 0;
	vel_x[0] = 0.0;
	vel_y[0] = 0.0;
	mass[0] = 3000;
}

int main()
{
	ofstream file, data;

	file.open("OpenCLTimings.csv");
	data.open("data.csv");
	file << "BODIES,CHUNKS,TIME" << endl;

	vector<int> bodies_vec = { 512 }; // , 1024, 5120};
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
				const unsigned int ITERS = 1000;												// Number of simulation iterations.
				const size_t LOCAL_WORK_SIZE = 512;												// Number of threads per block.
				const size_t GLOBAL_WORK_SIZE = ceil(N / LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE;	// Number of blocks required to satisfy N bodies with THREADS threads per block.
				const unsigned int ITER_CHUNKS = chunks;										// Number of chunks to seperate iterations into
				const unsigned int ITER_CHUNK_SIZE = ITERS / ITER_CHUNKS;						// Calculated size of iteration chunks

				// Status of OpenCl calls
				cl_int status;

				// Init OpenCL
				vector<cl_platform_id> platforms;
				vector<cl_device_id> devices;
				cl_context context;
				cl_command_queue cmd_queue;
				initialise_opencl(platforms, devices, context, cmd_queue);

				//print_opencl_info(devices);

				auto program = load_program("nbody.cl", context, devices[0], devices.size());
				auto kernel = clCreateKernel(program, "nbody", &status);

				// Declare host memory 
				float		 *host_pos_x;		// out
				float		 *host_pos_y;		// out
				float		 *host_vel_x;		// out
				float		 *host_vel_y;		// out
				float		 *host_mass;		// in
				unsigned int *host_n;			// in
				unsigned int *host_iters;		// in

				// Allocate host memory
				host_pos_x = (float*)malloc((sizeof(float) * N) * ITER_CHUNK_SIZE);
				host_pos_y = (float*)malloc((sizeof(float) * N) * ITER_CHUNK_SIZE);
				host_vel_x = (float*)malloc((sizeof(float) * N) * ITER_CHUNK_SIZE);
				host_vel_y = (float*)malloc((sizeof(float) * N) * ITER_CHUNK_SIZE);
				host_mass  = (float*)malloc((sizeof(float) * N));
				host_n     = (unsigned int*)malloc(sizeof(unsigned int));
				host_iters = (unsigned int*)malloc(sizeof(unsigned int));

				// Initialise host memory
				initBodies(host_pos_x, host_pos_y, host_vel_x, host_vel_y, host_mass, N);
				host_n[0] = N;
				host_iters[0] = ITER_CHUNK_SIZE;

				// Declare device memory
				cl_mem	dev_pos_x;			// out
				cl_mem	dev_pos_y;			// out
				cl_mem	dev_vel_x;			// out
				cl_mem	dev_vel_y;			// out
				cl_mem	dev_mass;			// in
				cl_mem	dev_n;				// in
				cl_mem  dev_iters;			// in
				
				//Allocate buffer size
				dev_pos_x = clCreateBuffer(context, CL_MEM_READ_WRITE, (sizeof(float) * N) * ITER_CHUNK_SIZE, nullptr, &status);
				dev_pos_y = clCreateBuffer(context, CL_MEM_READ_WRITE, (sizeof(float) * N) * ITER_CHUNK_SIZE, nullptr, &status);
				dev_vel_x = clCreateBuffer(context, CL_MEM_READ_WRITE, (sizeof(float) * N) * ITER_CHUNK_SIZE, nullptr, &status);
				dev_vel_y = clCreateBuffer(context, CL_MEM_READ_WRITE, (sizeof(float) * N) * ITER_CHUNK_SIZE, nullptr, &status);
				dev_mass  = clCreateBuffer(context, CL_MEM_READ_ONLY,  (sizeof(float) * N),					  nullptr, &status);
				dev_n     = clCreateBuffer(context, CL_MEM_READ_ONLY,  (sizeof(unsigned int)),				  nullptr, &status);
				dev_iters = clCreateBuffer(context, CL_MEM_READ_ONLY,  (sizeof(unsigned int)),				  nullptr, &status);

				auto start = system_clock::now();

				// Copy host data to device data
				status  = clEnqueueWriteBuffer(cmd_queue, dev_pos_x, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_pos_x, 0, nullptr, nullptr);
				status |= clEnqueueWriteBuffer(cmd_queue, dev_pos_y, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_pos_y, 0, nullptr, nullptr);
				status |= clEnqueueWriteBuffer(cmd_queue, dev_vel_x, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_vel_x, 0, nullptr, nullptr);
				status |= clEnqueueWriteBuffer(cmd_queue, dev_vel_y, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_vel_y, 0, nullptr, nullptr);
				status |= clEnqueueWriteBuffer(cmd_queue, dev_mass,  CL_TRUE, 0, (sizeof(float) * N),					host_mass,	0, nullptr, nullptr);
				status |= clEnqueueWriteBuffer(cmd_queue, dev_n,	 CL_TRUE, 0, (sizeof(unsigned int)),				host_n,		0, nullptr, nullptr);
				status |= clEnqueueWriteBuffer(cmd_queue, dev_iters, CL_TRUE, 0, (sizeof(unsigned int)),				host_iters, 0, nullptr, nullptr);

				if (status != CL_SUCCESS) {
					fprintf(stderr, "Write Buffer Failed\n");
					int a;
					cin >> a;
					return status;
				}

				status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_pos_x); 
				status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_pos_y); 
				status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_vel_x); 
				status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_vel_y); 
				status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_mass);	
				status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &dev_n);		
				status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &dev_iters); 

				if (status != CL_SUCCESS) {
					fprintf(stderr, "Kernal Arg Failed\n");
					int a;
					cin >> a;
					return status;
				}

				for (unsigned int i = 0; i < ITER_CHUNKS; i++)
				{

					// Enqueue the kernel for execution
					status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, nullptr);

					if (status != CL_SUCCESS) {
						fprintf(stderr, "Executing kernel failed\n");
						cout << status << endl;
						int a;
						cin >> a;
						return status;
					}

					clFinish(cmd_queue);

					// Read output buffer from GPU to Host mem
					clEnqueueReadBuffer(cmd_queue, dev_pos_x, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_pos_x, 0, nullptr, nullptr);
					clEnqueueReadBuffer(cmd_queue, dev_pos_y, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_pos_y, 0, nullptr, nullptr);
					clEnqueueReadBuffer(cmd_queue, dev_vel_x, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_vel_x, 0, nullptr, nullptr);
					clEnqueueReadBuffer(cmd_queue, dev_vel_y, CL_TRUE, 0, (sizeof(float) * N) * ITER_CHUNK_SIZE, host_vel_y, 0, nullptr, nullptr);

					// Write to file
					for (int i = 0; i < ITER_CHUNK_SIZE; i++)
					{
						for (int k = 0; k < N; k++)
						{
							unsigned int j = k + (i * N);

							data << i << "," << host_pos_x[j] << "," << host_pos_y[j] << "," << host_vel_x[j] << "," << host_vel_y[j] << "," << host_mass[j] << endl;

						}
					}
					
				}

				data.close();

				auto end = system_clock::now();
				auto total = end - start;
				cout << "Bodies: " << N << " Chunks: " << ITER_CHUNKS << " Time Taken: " << duration_cast<milliseconds>(total).count() << "ms" << endl;
				file << duration_cast<milliseconds>(total).count() << ",";

				// Free mallocs
				free(host_pos_x);
				free(host_pos_y);
				free(host_vel_x);
				free(host_vel_y);
				free(host_mass);
				free(host_n);
				free(host_iters);

				// Free OpenCL resources
				clReleaseMemObject(dev_pos_x);
				clReleaseMemObject(dev_pos_y);
				clReleaseMemObject(dev_vel_x);
				clReleaseMemObject(dev_vel_y);
				clReleaseMemObject(dev_mass);
				clReleaseMemObject(dev_n);
				clReleaseMemObject(dev_iters);
				clReleaseCommandQueue(cmd_queue);
				clReleaseContext(context);
				clReleaseKernel(kernel);
				clReleaseProgram(program);
			}

			file << endl;
		}
	}

	int a;
	cin >> a;

	return 0;
}

