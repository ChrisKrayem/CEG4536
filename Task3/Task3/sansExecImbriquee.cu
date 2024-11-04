%% writefile task3_4.cu

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Structure pour l'op�ration d'addition
template <typename T>
struct Add {
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

// Kernel de r�duction parall�le sans ex�cution imbriqu�e (Parallel reduction kernel without nested execution)
template <typename T>
__global__ void reductionKernel(T* input, T* output, int n) {
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Charger les �l�ments dans la m�moire partag�e (Load elements into shared memory)
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    Add<T> op;

    // R�duction avec d�roulage par 4 (Reduction with loop unrolling by 4)
    for (int s = blockDim.x / 4; s > 32; s /= 4) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
            sdata[tid] = op(sdata[tid], sdata[tid + s * 2]);
            sdata[tid] = op(sdata[tid], sdata[tid + s * 3]);
        }
        __syncthreads();
    }

    // R�duction par warp (Warp reduction)
    if (tid < 32) {
        volatile int* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Le premier thread stocke le r�sultat partiel (First thread stores the partial result)
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int size = 128;  // Taille du tableau d'entr�e
    int blockSize = 32;  // Taille du bloc
    int gridSize = (size + blockSize - 1) / blockSize;  // Nombre de blocs

    int* h_input = (int*)malloc(size * sizeof(int));
    int* h_output = (int*)malloc(gridSize * sizeof(int));
    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, gridSize * sizeof(int));

    // Initialisation du tableau d'entr�e
    for (int i = 0; i < size; i++) h_input[i] = i + 1;

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Lancement du kernel pour la r�duction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reductionKernel << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds);

    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculer la somme totale
    int totalSum = 0;
    for (int i = 0; i < gridSize; i++) totalSum += h_output[i];
    printf("Somme Totale: %d\n", totalSum);

    // Lib�ration de la m�moire
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}