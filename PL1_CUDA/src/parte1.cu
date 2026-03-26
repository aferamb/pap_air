#include "parte1.cuh"

#include <stdio.h>
#include <iostream>

#include "comun.cuh"

namespace {

/*
    El umbral firmado se interpreta igual que en el enunciado:

    - positivo para retrasos;
    - negativo para adelantos.
*/
__device__ bool matchesSignedThreshold(int value, int threshold)
{
    if (threshold >= 0) {
        return value >= threshold;
    }

    return value <= threshold;
}

/*
    Cada hilo procesa una fila de DEP_DELAY.
*/
__global__ void phase1DepartureDelayKernel(const float* delayValues, int totalElements, int threshold)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalElements) {
        return;
    }

    const float rawValue = delayValues[idx];

    if (rawValue != rawValue) {
        return;
    }

    const int delayValue = static_cast<int>(rawValue);

    if (matchesSignedThreshold(delayValue, threshold)) {
        const char* label = threshold >= 0 ? "Retraso" : "Adelanto";
        printf("- Hilo #%d: %s de %d minutos\n", idx, label, delayValue);
    }
}

} // namespace

void phase01(int threshold)
{
    const LaunchConfig launchConfig = computeLaunchConfig(g_rowCount);

    std::cout << "DEP_DELAY | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    phase1DepartureDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        d_depDelay,
        g_rowCount,
        threshold);

    if (!executeAndWait("phase1DepartureDelayKernel")) {
        std::cout << "La Fase 01 no se ha podido completar.\n";
    }
}
