// OpenCLIntro.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>

using namespace std;

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

int main()
{
	// Status of OpenCl calls
	cl_int status;

	// Init OpenCL
	vector<cl_platform_id> platforms;
	vector<cl_device_id> devices;
	cl_context context;
	cl_command_queue cmd_queue;
	initialise_opencl(platforms, devices, context, cmd_queue);

	print_opencl_info(devices);

	auto program = load_program("simply_multiply.cl", context, devices[0], devices.size());
	auto kernel = clCreateKernel(program, "simply_multiply", &status);

	//Num of elements and size of GPU buffer
	const unsigned int elements = 2048;
	const unsigned int matrix_size = 64;
	const unsigned int data_size = sizeof(int) * (matrix_size * matrix_size);

	// Host data - stored in main memory
	array<int, matrix_size * matrix_size> A;
	array<int, matrix_size * matrix_size> B;
	array<int, matrix_size * matrix_size> C;

	unsigned int matrix_pos = 0;

	//Init the host data
	for (unsigned int i = 0; i < matrix_size; ++i)
	{
		for (unsigned int j = 0; i < matrix_size; ++i)
		{
			A[i] = B[i] = 1;

			matrix_pos++;
		}
	}

	//Create device buffers - this goes on the GPU
	cl_mem buffer_A; // Input array on device
	cl_mem buffer_B; // Input array on device
	cl_mem buffer_C; // Input array on device
	//Allocate buffer size
	buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, nullptr, &status);
	buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, nullptr, &status);
	buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);

	// Copy host data to device data
	status = clEnqueueWriteBuffer(cmd_queue, buffer_A, CL_FALSE, 0, data_size, A.data(), 0, nullptr, nullptr);
	status = clEnqueueWriteBuffer(cmd_queue, buffer_B, CL_FALSE, 0, data_size, B.data(), 0, nullptr, nullptr);

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_C); //Output
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_A); //Matrix A
	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &buffer_B); //Matrix B
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &matrix_size); //Width A
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &matrix_size); //Height A
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &matrix_size); //Width B
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &matrix_size); //Height B

	// Configure work dimensions - 1D of elements
	array<size_t, 1> global_work_size = { matrix_size * matrix_size };

	// Enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, global_work_size.data(), nullptr, 0, nullptr, nullptr);

	// Read output buffer from GPU to Host mem
	clEnqueueReadBuffer(cmd_queue, buffer_C, CL_TRUE, 0, data_size, C.data(), 0, nullptr, nullptr);

	// Verify the output
	auto result = true;
	int i = 0;
	// Iterate through each value in result array
	for (auto &e : C)
	{
		// Check value
		if (e != matrix_size)
		{
			result = false;
			break;
		}
		++i;
	}

	if (result)
		cout << "Output correct!" << endl;
	else
		cout << "Output incorrect!" << endl;


	// Free OpenCL resources
	clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_B);
	clReleaseMemObject(buffer_C);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	int a;
	cin >> a;

	return 0;
}

