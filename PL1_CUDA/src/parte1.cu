#include "parte1.cuh"

#include <stdio.h>
#include <iostream>

#include "comun.cuh"

namespace {

/**
 * @brief Determina si un valor de retraso o adelanto cumple con el umbral dado, considerando que el umbral puede ser positivo (retraso) o negativo (adelanto).
 * 
 * @param value Valor de retraso o adelanto a evaluar.
 * @param threshold umbral que puede ser positivo (para retrasos) o negativo (para adelantos).  
 * @return __device__ 
 */
__device__ bool matchesSignedThreshold(int value, int threshold)
{
    if (threshold >= 0) {
        return value >= threshold;
    }

    return value <= threshold;
}

/**
 * @brief Kernel de CUDA para evaluar los retrasos de salida (DEP_DELAY) en función de un umbral dado. 
 * El kernel imprime los valores que cumplen con el umbral, indicando si se trata de un retraso o un adelanto.
 * 
 * @param delayValues Puntero a la columna de retrasos de salida en la GPU.
 * @param totalElements Número total de elementos en la columna de retrasos de salida.
 * @param threshold Umbral que puede ser positivo (para retrasos) o negativo (para adelantos). El kernel evaluará los valores en función de este umbral.
 * @return __global__ 
 */
__global__ void phase1DepartureDelayKernel(const float* delayValues, int totalElements, int threshold)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // Cálculo del índice global del hilo (la formulita)

    if (idx >= totalElements) {
        return;
    }

    const float rawValue = delayValues[idx];

    if (rawValue != rawValue) {
        return;
    }

    const int delayValue = static_cast<int>(rawValue);

    if (matchesSignedThreshold(delayValue, threshold)) {
        const char* label = threshold >= 0 ? "Retraso" : "Adelanto"; // Determina si el valor cumple con el umbral de retraso o adelanto y asigna la etiqueta correspondiente.
        printf("- Hilo #%d: %s de %d minutos\n", idx, label, delayValue); // Imprime el índice del hilo, el tipo de evento (retraso o adelanto) y el valor del retraso o adelanto que cumple con el umbral.
    }
}

} // namespace

/**
 * @brief Ejecuta la Fase 01 del programa, que evalúa los retrasos de salida en función de un umbral dado.
 * 
 * @param threshold Umbral que puede ser positivo (para retrasos) o negativo (para adelantos).
 */
void phase01(int threshold)
{
    const LaunchConfig launchConfig = computeLaunchConfig(g_rowCount);

    std::cout << "DEP_DELAY | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    // Lanzar el kernel de CUDA para evaluar los retrasos de salida en función del umbral dado. El kernel imprimirá los valores que cumplen con el umbral, indicando si se trata de un retraso o un adelanto.
    phase1DepartureDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        d_depDelay,
        g_rowCount,
        threshold);

    if (!executeAndWait("phase1DepartureDelayKernel")) {
        std::cout << "La Fase 01 no se ha podido completar.\n";
    }
}
