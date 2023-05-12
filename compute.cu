#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
//__global__ void compute_help()
//{
//}
__global__ void initialize(int NUMENTITIES, vector3 *values, vector3 **accels)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < NUMENTITIES; i += stride)
	{
		accels[i] = &values[i * NUMENTITIES];
	}
}

__global__ void fill(int NUMENTITIES, vector3 *values, vector3 **accels)
{
	for (int i = index; i < NUMENTITIES; i += stride)
	{
		// first compute the pairwise accelerations.  Effect is on the first argument.
		for (int j = 0; j < NUMENTITIES; j++)
		{
			if (i == j)
			{
				FILL_VECTOR(accels[i][j], 0, 0, 0);
			}
			else
			{
				vector3 distance;
				for (k = 0; k < 3; k++)
					distance[k] = hPos[i][k] - hPos[j][k];
				double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
				double magnitude = sqrt(magnitude_sq);
				double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
				FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
			}
		}
	}
}
void compute()
{
	// make an acceleration matrix which is NUMENTITIES squared in size;
	int i, j, k;
	vector3 *values = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3 **accels = (vector3 **)malloc(sizeof(vector3 *) * NUMENTITIES);
	vector3 *d_values;
	cudaMalloc((vector3 **)&d_values, sizeof(values));
	vector3 **d_accels;
	cudaMalloc((vector3 ***)&d_accels, sizeof(accels));

	int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;
	construct_row<<<1, 256>>>(NUMENTITIES, d_values, d_accels);
	cudaDeviceSynchronize();

	// sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i = 0; i < NUMENTITIES; i++)
	{
		vector3 accel_sum = {0, 0, 0};
		for (j = 0; j < NUMENTITIES; j++)
		{
			for (k = 0; k < 3; k++)
				accel_sum[k] += accels[i][j][k];
		}
		// compute the new velocity based on the acceleration and time interval
		// compute the new position based on the velocity and time interval
		for (k = 0; k < 3; k++)
		{
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}
	}
	free(accels);
	free(values);
	cudaFree(d_accels);
	cudaFree(d_values);
}
