#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>
#include <iostream>
using namespace std;

// Fonction pour transposer une matrice sur l'hôte
void transposeMatrix(int* src, int* dst, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dst[j * N + i] = src[i * N + j];
        }
    }
}

// Kernel pour multiplication matricielle avec B transposé
__global__ void matrixMultiplyWithTransposedB(int* A, int* B_T, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Vérifier les limites
    if (row < N && col < N) {
        int tmp = 0;

        // Produit scalaire de la ligne de A et de la ligne de B_T (qui est la colonne de B)
        for (int i = 0; i < N; i++) {
            tmp += A[row * N + i] * B_T[col * N + i];
        }

        // Stocker le résultat
        C[row * N + col] = tmp;
    }
}

// Fonction principale pour effectuer la multiplication matricielle optimisée
void matrixMultiplyOptimized(int* h_A, int* h_B, int* h_C, int N) {
    size_t size = N * N * sizeof(int);

    // Allocation mémoire hôte pour la transposée de B
    int* h_B_T = (int*)malloc(size);

    // Calcul de la transposée de B
    transposeMatrix(h_B, h_B_T, N);

    // Allocation mémoire sur le GPU
    int* d_A, * d_B_T, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B_T, size);
    cudaMalloc((void**)&d_C, size);

    // Copie des matrices sur le GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_T, h_B_T, size, cudaMemcpyHostToDevice);

    // Définir les dimensions des blocs et des grilles
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Lancer le kernel
    matrixMultiplyWithTransposedB << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B_T, d_C, N);
    cudaDeviceSynchronize();

    // Copier le résultat du GPU vers l'hôte
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Libération de la mémoire
    cudaFree(d_A);
    cudaFree(d_B_T);
    cudaFree(d_C);
    free(h_B_T);
}

// Fonction pour vérifier les résultats
void verify_result(int* A, int* B, int* C, int N) {
    int tmp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += A[i * N + k] * B[k * N + j];
            }
            assert(tmp == C[i * N + j]);
        }
    }
}

int main() {
    int N = 4;  // Taille des matrices (NxN) - ajustez selon vos besoins
    size_t size = N * N * sizeof(int);

    // Allocation mémoire hôte
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

    // Initialisation des matrices A et B avec des valeurs arbitraires
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i + 1;  // Exemple : valeurs séquentielles
        h_B[i] = i + 1;  // Exemple : valeurs séquentielles
    }

    // Mesure du temps d'exécution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Multiplication matricielle optimisée
    matrixMultiplyOptimized(h_A, h_B, h_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Temps d'exécution du kernel : %f ms\n", milliseconds);

    // Affichage de la matrice résultat
    printf("Matrice résultante C :\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Vérification des résultats
    verify_result(h_A, h_B, h_C, N);
    cout << "Multiplication matricielle calculée avec succès sur le GPU et vérifiée par le CPU." << endl;

    // Libération de la mémoire hôte
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

