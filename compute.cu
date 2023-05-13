#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void computePairAccels(vector3 **accels, vector3 *values, vector3 *hPos, vector3 *hVel, double *mass) {
	int idx = threadIdx.x;
	int stride = blockDim.x;
	int i, j, k;
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=idx;i<NUMENTITIES;i+=stride){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
}

__global__ void computeEffects(vector3 **accels, vector3 *hPos, vector3 *hVel){
	int idx = threadIdx.x;
	int stride = blockDim.x;
	int i, j, k;

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=idx;i<NUMENTITIES;i+=stride){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	int blockSize = 128;
	int numBlocks = (NUMENTITIES - 1) / blockSize + 1; 
	
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

	// DEVICE VARIABLE DECLARATIONS
	vector3* d_values;
	vector3** d_accels;
	cudaMalloc((void**)&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc((void**)&d_accels, sizeof(vector3*)*NUMENTITIES);
	
	// GLOBAL DEVICE DECLARATIONS
	cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**)&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);


	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	cudaMemcpy(d_values, values, sizeof(vector3)*NUMENTITIES*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accels, accels, sizeof(vector3*)*NUMENTITIES, cudaMemcpyHostToDevice);
	
	computePairAccels<<<numBlocks, blockSize>>>(d_accels, d_values, d_hPos, d_hVel, d_mass);
	cudaDeviceSynchronize();
	computeEffects<<<numBlocks, blockSize>>>(d_accels, d_hPos, d_hVel);
	cudaDeviceSynchronize();
	cudaMemcpy(d_values, values, sizeof(vector3)*NUMENTITIES*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_accels, accels, sizeof(vector3*)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaFree(d_values);
	cudaFree(d_accels);
	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
	free(accels);
	free(values);
}