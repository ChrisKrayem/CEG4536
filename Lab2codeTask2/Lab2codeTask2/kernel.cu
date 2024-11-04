#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Structure for addition operation
template <typename T>
struct Add {
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

// Structure for multiplication operation
template <typename T>
struct Multiply {
    __device__ T operator()(T a, T b) const {
        return a * b;
    }
};

// Kernel with loop unrolling for reduction with templated input and output
template <typename T, typename Op>
__global__ void parallelReduceUnrolled(T* input, T* output, int size, Op op, T neutralElement) {
    extern __shared__ T sharedData[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, with bounds checking
    if (index < size) sharedData[tid] = input[index];
    else sharedData[tid] = neutralElement;

    __syncthreads();

    // Perform reduction with loop unrolling by 4
    for (int s = blockDim.x / 4; s > 32; s /= 4) {
        if (tid < s) {
            sharedData[tid] = op(sharedData[tid], sharedData[tid + s]);
            sharedData[tid] = op(sharedData[tid], sharedData[tid + s * 2]);
            sharedData[tid] = op(sharedData[tid], sharedData[tid + s * 3]);
        }
        __syncthreads();
    }

    // Réduction par warp (Warp reduction)
    if (tid < 32) {
        volatile int* vsmem = sharedData;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write the result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

int main() {
    int size = 128;
    int blockSize = 32;
    int gridSize = (size + blockSize - 1) / blockSize;

    int* h_input = (int*)malloc(size * sizeof(int));
    int* h_output = (int*)malloc(gridSize * sizeof(int));
    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, gridSize * sizeof(int));

    // Initialize the input array
    for (int i = 0; i < size; i++) h_input[i] = i + 1;

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel for addition using unrolled reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    parallelReduceUnrolled<int, Add<int>> << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size, Add<int>(), 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < gridSize; i++) sum += h_output[i];
    printf("Sum: %d\n", sum);

    // Launch the kernel for multiplication using unrolled reduction
    /*parallelReduceUnrolled<int, Multiply<int>> << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size, Multiply<int>(), 1);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int product = 1;
    for (int i = 0; i < gridSize; i++) product *= h_output[i];
    printf("Product: %d\n", product);*/

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
