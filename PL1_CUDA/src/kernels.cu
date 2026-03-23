// kernels.cu
// Implementación de kernels GPU

#include <stdio.h>
#include "kernels.cuh"

// Kernel de prueba que imprime el ID de cada hilo
__global__ void holaGPU() {
    printf("Hola desde GPU hilo %d\n", threadIdx.x);
}

// FASE 01: Kernel de retraso en despegues
// Cada hilo revisa su posición en el vector dep_delay
// Si el retraso supera el umbral, imprime el índice y el valor
__global__ void depDelayKernel(float* dep_delay, size_t n, float umbral) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) { // Validar que idx esté dentro del vector
        float delay = dep_delay[idx];

        // Comparar con umbral
        if (delay >= umbral) {
            printf("Vuelo %d tiene retraso de %.1f minutos\n", idx, delay);
        }
    }
}