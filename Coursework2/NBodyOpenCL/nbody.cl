__kernel void simply_multiply(__global__ float* pos_x, __global__ float* pos_y, __global__ float* vel_x, __global__ float* vel_y, __global__ float* mass, __global__ unsigned int* n, __global__ unsigned int* iterations)
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

				float dx = pos[body1].x - pos[body2].x;
				float dy = pos[body1].y - pos[body2].y;
				float distance = sqrt(dx*dx + dy*dy + DAMPENING);
				
				float force = G * (mass[body1] * (mass[body2] / distance));

				fx += force * (dx / distance);
				fy += force * (dy / distance);
			}

			vel[body2].x += TIME_STEP * (fx / mass[body2]);
			vel[body2].y += TIME_STEP * (fy / mass[body2]);

			pos[body2].x += TIME_STEP * vel[body2].x;
			pos[body2].y += TIME_STEP * vel[body2].y;

			__syncthreads();

			offset = -1;
		}
	}
}