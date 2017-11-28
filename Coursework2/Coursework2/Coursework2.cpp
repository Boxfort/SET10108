#include "stdafx.h"
#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

typedef struct { double x, y, vx, vy, mass; } Body;

const unsigned int N = 5120;
const unsigned int ITERS = 1000;
const double G = 6.674e-11;
const double TIME_STEP = 1;
const double DAMPENING = 1e-9;
const double TOLERANCE = 0.01;
const double PI = 3.14159265358979323846;


void initBodies(Body* bodies)
{
	for (int i = 1; i < N; i++)
	{
		bodies[i].x = 2.0f * (rand() / (double)RAND_MAX) - 1.0f; // Init at position between -1 : 1
		bodies[i].y = 2.0f * (rand() / (double)RAND_MAX) - 1.0f; // Init at position between -1 : 1
		bodies[i].vx = 0;
		bodies[i].vy = 0;
		bodies[i].mass = rand() % (500 - 100 + 1) + 100;
	}

	bodies[0].x = 0;
	bodies[0].y = 0;
	bodies[0].vx = 0.0;
	bodies[0].vy = 0.0;
	bodies[0].mass = 3000;
}

void calculateForces(Body* bodies)
{
	for (int i = 0; i < N; i++) 
	{
		double fx = 0.0, fy = 0.0;

		for (int j = 0; j < N; j++) 
		{
			if (i == j) { continue; }

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

/*
void mergeBodies(Body* bodies)
{
	for (int i = 0; i < N; i++)
	{
		if (bodies[i].dead) { continue; }

		for (int j = 0; j < N; j++)
		{
			if (i == j) { continue; }
			if (bodies[j].dead) { continue; }

			if ((bodies[i].x >= (bodies[j].x - TOLERANCE) && bodies[i].x <= bodies[j].x + TOLERANCE) && (bodies[i].y >= (bodies[j].y - TOLERANCE) && bodies[i].y <= (bodies[j].y + TOLERANCE) ))
			{
				cout << "collided x1"<< bodies[i].x << " x2 " << bodies[j].x << endl;
				// Colision detected kinda lol
				if (bodies[i].mass >= bodies[j].mass)
				{
					bodies[i].mass += bodies[j].mass;
					bodies[j].dead = true;
					bodies[j].mass = 0;
				}
				else
				{
					bodies[j].mass += bodies[i].mass;
					bodies[i].dead = true;
					bodies[i].mass = 0;
				}
			}
		}
	}
}
*/

int main()
{
	srand(time(NULL));

	int dataSize = sizeof(Body) * N;   // Total size of data
	Body *p = (Body*)malloc(dataSize); // Pointer to array of bodies

	ofstream file, timings;

	file.open("data.csv");
	timings.open("SequentialTiming.csv");

	unsigned int timing_iterations = 50;

	for (int t = 0; t < timing_iterations; t++)
	{
		initBodies(p);

		auto start = system_clock::now();

		for (unsigned int i = 0; i < ITERS; i++)
		{
			calculateForces(p);

			//for (int j = 0; j < N; j++)
			//{
			//	file << i << "," << p[j].x << "," << p[j].y << "," << p[j].vx << "," << p[j].vy << "," << p[j].mass << endl;
			//}

			//auto elapsed = system_clock::now() - start;
			//auto remaining = duration_cast<seconds>((elapsed / (i+1)) * (ITERS - i + 1)).count();

			cout << t << " | Iteration " << i << " of " << ITERS << endl; // " Time Remaining: " << remaining << " Seconds." << "\r";
		}

		auto end = system_clock::now();
		timings << duration_cast<milliseconds>(end - start).count() << ",";
	}

	file.close();
	timings.close();

	return 0;
}


