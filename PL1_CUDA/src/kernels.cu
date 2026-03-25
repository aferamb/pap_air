#include "kernels.cuh"

#include <limits.h>

/*
    Kernel:
    - Cada hilo procesa un elemento
    - Usa operaciones at�micas globales
*/

__global__ void reductionSimple(int* data, int* result, int n, bool isMax)
{
    // Formula clasica de acceso linealizado 1D en CUDA.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Si sobran hilos respecto al tamano real del vector, esos hilos salen
    // inmediatamente para no leer fuera de rango.
    if (idx >= n) {
        return;
    }

    // Cada hilo solo lee el elemento que le corresponde dentro del vector.
    const int value = data[idx];

    // La rama selecciona el tipo de reduccion pedido por el host. Ambas usan
    // operaciones atomicas porque varios hilos pueden intentar actualizar el
    // mismo acumulador global al mismo tiempo.
    if (isMax) {
        atomicMax(result, value);
    } else {
        atomicMin(result, value);
    }
}
