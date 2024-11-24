#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//ajoute/
#include <cassert>
#include <iostream>
using namespace std;

__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

        //boundary check
    if (row < N && col < N) {
        int tmp = 0; 

        //calcul d'une position/terme par threads de la multiplication matricielle
        for (int i = 0; i < N; i++) {
            tmp += A[row * N + i] * B[i * N + col];
        }
        //WriteBack le resultat
        C[row * N + col] = tmp;
    }

}


// Host function to allocate memory and invoke the kernel
void matrixMultiply(int* h_A, int* h_B, int* h_C, int N) {
    size_t size = N * N * sizeof(int);

    // Allocate memory on the device
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);  // Enough blocks to cover the matrix

    // Launch the kernel
    matrixMultiplyKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

//verif on CPU
void verify_result(int* A, int* B, int* C, int N) {
    int tmp;
    // for every row
    for (int i = 0; i < N; i++) {
        //for every col
        for (int j = 0; j < N; j++) {
            //for every element in the row-col pair
            tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += A[i * N + k] * B[k * N + j];
            }

            //check each result
            assert(tmp == C[i * N + j]);
        }
    }
}

int main() {
    int N = 4;  // Size of the matrices (NxN) - Adjust as needed
    size_t size = N * N * sizeof(int);

    // Allocate host memory
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i + 1;  // Example: sequential values
        h_B[i] = i + 1;  // Example: sequential values
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Perform matrix multiplication
    matrixMultiply(h_A, h_B, h_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds);

    // Print the result matrix C
    printf("Resultant Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    //verif result with the CPU
    verify_result(h_A, h_B, h_C, N);
    cout << "Matrix Multiplication Successfully Calculated on GPU and Verified by the CPU" << endl;

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
