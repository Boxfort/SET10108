__constant float DAMPENING = 1e-9;
__constant float TIME_STEP = 1.0;
__constant float G = 6.674e-11;

__kernel void nbody(__global float* pos_x, __global float* pos_y, __global float* vel_x, __global float* vel_y, __global float* mass, __global unsigned int* n, __global unsigned int* iterations)
{
	// Calculate index
	int idx = get_global_id(0);

	float fx = 0.0, fy = 0.0;
	int offset = 0;

	// Ensure the extra threads asigned are not run to prevent heap corruption.
	if (idx < n[0])
	{
		for (int i = 0; i < iterations[0]; i++)
		{
			int body2 = idx + ((i - offset) * n[0]);

			// For each body
			for (int j = 0; j < n[0]; j++)
			{
				//if (idx == j) { continue; }

				int body1 = j + ((i - offset) * n[0]);

				float dx = pos_x[body1] - pos_x[body2];
				float dy = pos_y[body1] - pos_y[body2];
				float distance = sqrt(dx*dx + dy*dy + DAMPENING);
				
				float force = G * (mass[body1] * (mass[body2] / distance));

				fx += force * (dx / distance);
				fy += force * (dy / distance);
			}

			vel_x[body2] += TIME_STEP * (fx / mass[body2]);
			vel_y[body2] += TIME_STEP * (fy / mass[body2]);

			pos_x[body2] += TIME_STEP * vel_x[body2];
			pos_y[body2] += TIME_STEP * vel_y[body2];

			offset = -1;
		}
	}
}