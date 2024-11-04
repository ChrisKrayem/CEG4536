%% writefile task3_3.cu

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Structure pour l'opération d'addition
template <typename T>
struct Add {
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

// Kernel de réduction parallèle avec exécution imbriquée (Parallel reduction kernel with nested execution)
template <typename T>
__global__ void nestedReduction(T* input, T* output, int n) {
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Charger deux éléments par thread pour réduire les accès mémoire (Load two elements per thread to reduce memory accesses)
    sdata[tid] = (i < n ? input[i] : 0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    __syncthreads();

    Add<T> op;

    // Réduction avec déroulage par 4 (Reduction with loop unrolling by 4)
    for (int s = blockDim.x / 4; s > 32; s /= 4) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
            sdata[tid] = op(sdata[tid], sdata[tid + s * 2]);
            sdata[tid] = op(sdata[tid], sdata[tid + s * 3]);
        }
        __syncthreads();
    }

    // Réduction par warp (Warp reduction)
    if (tid < 32) {
        volatile int* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }



    // Le premier thread stocke le résultat partiel (First thread stores the partial result)
    if (tid == 0) output[blockIdx.x] = sdata[0];

    // Lancer un nouveau kernel si nécessaire pour finaliser la réduction (Launch a new kernel if necessary to complete the reduction)
    if (blockIdx.x == 0 && tid == 0) {
        // Si le nombre de blocs est supérieur à 1, lancer une réduction imbriquée (If more than 1 block, launch nested reduction)
        int numBlocks = (n + (blockDim.x * 2 - 1)) / (blockDim.x * 2);
        if (numBlocks > 1) {
            nestedReduction << < numBlocks, blockDim.x, blockDim.x * sizeof(T) >> > (output, output, numBlocks);
        }
    }
}

int main() {
    int size = 128;  // Taille du tableau d'entrée
    int blockSize = 32;  // Taille du bloc
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);  // Nombre de blocs

    int* h_input = (int*)malloc(size * sizeof(int));
    int* h_output = (int*)malloc(gridSize * sizeof(int));
    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, gridSize * sizeof(int));

    // Initialisation du tableau d'entrée
    for (int i = 0; i < size; i++) h_input[i] = i + 1;

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Lancement du kernel pour la réduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    nestedReduction << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);
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

    // Libération de la mémoire
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
