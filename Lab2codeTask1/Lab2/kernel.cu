
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void parallelAdd(int* input, int* output, int size) {
	extern __shared__ int sharedData[];
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Charger les données en mémoire partagée
	if (index < size) sharedData[tid] = input[index];
	else sharedData[tid] = 0; // Multiplication donc initialisé à l 

	__syncthreads();

	// Réduction parallèle (multiplication)
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sharedData[tid] += sharedData[tid + s];
		}
		__syncthreads();
	}
	// Écrire le résultat dans la mémoire globale 
	if (tid == 0) output[blockIdx.x] = sharedData[0];
}

int main() {

	int size = 128;
	int blockSize = 4;
	int gridSize = (size + blockSize - 1) / blockSize;

	int * h_input = (int*) malloc(size * sizeof(int));
	int * h_output = (int *)malloc(gridSize * sizeof(int));
	int * d_input,  * d_output;
	cudaMalloc((void**)&d_input, size * sizeof(int));
	cudaMalloc((void**)&d_output, gridSize * sizeof(int));

	// Initialisation du tableau avec des valeurs non nulles
	for (int i = 0; i < size; i++) h_input[i] = i + 1; // Vous pouvez changer les valeurs ici

	cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	parallelAdd << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel Execution Time: %f ms\n", milliseconds);
	cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

	int sum = 0; 
	for (int i = 0; i < gridSize; i++) sum += h_output[i];
	printf("Sum: %d\n", sum);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);

	return 0;
}