#pragma once

/*
    Kernel de reducción simple (Fase 03 - Variante 3.1)

    Cada hilo:
    - Lee una posición del array
    - Aplica atomicMax o atomicMin sobre un valor global
*/

__global__ void reductionSimple(int* data, int* result, int n, bool isMax);