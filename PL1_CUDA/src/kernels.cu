#include "kernels.cuh"
#include <limits.h>

/*
    Kernel:
    - Cada hilo procesa un elemento
    - Usa operaciones atómicas globales
*/

__global__ void reductionSimple(int* data, int* result, int n, bool isMax)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    int value = data[idx];

    if (isMax)
        atomicMax(result, value);
    else
        atomicMin(result, value);
}