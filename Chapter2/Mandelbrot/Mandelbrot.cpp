// Mandelbrot.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "lib/FreeImage.h"
#include <vector>
#include <thread>
#include <future>
#include <iostream>
#include <string>

using namespace std;

// Number of iterations to find each pixel value.
const unsigned int MAX_ITERATIONS = 1000;
// Dimension of the image to generate (in pixels)
const unsigned int DIMENSION = 1024; //32768;

const unsigned int SPLIT_AMOUNT = 4;

const int BPP = 24;

// Mandelbrot dimesnions are ([-2.1, 1.0], [-1.3, 1.3])
const double X_MIN = -2.1;
const double X_MAX = 1.0;
const double Y_MIN = -1.3;
const double Y_MAX = 1.3;

// Convert from mandlebrot coordinate to image coordinate
const double INTEGRAL_X = (X_MAX - X_MIN) / static_cast<double>(DIMENSION);
const double INTEGRAL_Y = (Y_MAX - Y_MIN) / static_cast<double>(DIMENSION);

vector<double> mandelbrot(unsigned int start_y, unsigned int end_y, unsigned int start_x, unsigned int end_x)
{
	// Declare the values that we will use
	double x, y, x1, y1, xx = 0.0;
	unsigned int loop_count = 0;
	
	vector<double> results;

	// Loop through each line

	y = Y_MIN + (start_y * INTEGRAL_Y);
	for (unsigned int y_coord = start_y; y_coord < end_y; ++y_coord)
	{
		x = X_MIN;
		// Loop through each pixel on the line.
		for (unsigned int x_coord = start_x; x_coord < end_x; ++x_coord)
		{
			x1 = 0.0, y1 = 0.0;
			loop_count = 0;

			while (loop_count < MAX_ITERATIONS && sqrt((x1 * x1) + (y1 * y1)) < 2.0)
			{
				++loop_count;
				xx = (x1 * x1) - (y1 * y1) + x;
				y1 = 2 * x1 * y1 + y;
				x1 = xx;
			}
			// Get Value after loop has completed
			double val = static_cast<double>(loop_count) / static_cast<double>(MAX_ITERATIONS);

			// Push value to results
			results.push_back(val);
			// Increase x based on integral
			x += INTEGRAL_X;
		}
		y += INTEGRAL_Y;
	}

	return results;
}

void save_image(vector<vector<double>>* image_data, unsigned int dimension,  const char* filename)
{
	cout << "Creating file: " << filename << ".png" << endl;

	FreeImage_Initialise();

	FIBITMAP* bitmap = FreeImage_Allocate(dimension, dimension, BPP);
	RGBQUAD color;

	if (!bitmap)
		exit(1);

	vector<double> pixel_data;

	for (auto &v : *image_data)
	{
		for (auto &d : v)
		{
			pixel_data.push_back(d);
		}
	}

	for (int x = 0; x < dimension; x++)
	{
		for (int y = 0; y < dimension; y++)
		{
			
			color.rgbRed = pixel_data[x + (y * dimension)] * 255;
			color.rgbBlue = pixel_data[x + (y * dimension)] * 255;
			color.rgbGreen = pixel_data[x + (y * dimension)] * 255;
			FreeImage_SetPixelColor(bitmap, x, y, &color);
		}
	}


	if (FreeImage_Save(FIF_PNG, bitmap, filename, 0))
	{
		cout << "Image saved." << endl;
	}

	FreeImage_DeInitialise();
}

int main()
{
	// Get the number of hardware threads
	auto num_threads = thread::hardware_concurrency();

	auto sub_dimension = DIMENSION / SPLIT_AMOUNT;

	for (int i = 0; i < SPLIT_AMOUNT; i++)
	{
		for (int j = 0; j < SPLIT_AMOUNT; j++)
		{
			int start_x = sub_dimension * i;
			int end_x = (i + 1) * sub_dimension;
			int start_y = sub_dimension * j;
			int end_y = (j + 1) * sub_dimension;

			// Determine strip height
			auto strip_height = sub_dimension / num_threads;

			// Create futures
			vector<future<vector<double>>> futures;
			for (unsigned int i = 0; i < num_threads; ++i)
			{
				//Range is used to determine number of values to process
				futures.push_back(async(mandelbrot, (i * strip_height) + start_y, ((i + 1) * strip_height) + start_y, start_x, end_x));
			}

			vector<vector<double>> results;

			for (auto &f : futures)
			{
				results.push_back(f.get());
			}

			string filename = "Mandelbrot" + to_string(i) + to_string(j) + ".png";

			save_image(&results, sub_dimension,filename.c_str());

			results.clear();
		}
	}

    return 0;
}

