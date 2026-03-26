#include "parte2.cuh"

#include <stdio.h>

#include <iostream>
#include <vector>

#include "comun.cuh"

namespace {

// Umbral compartido por todos los hilos de la Fase 02 (memoria constante mejor ). Compartir e vivir.
__constant__ int d_phase2Threshold;

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
 * @brief Copia el umbral a la memoria constante de la GPU para que esté disponible para todos los hilos durante la ejecución del kernel.
 * 
 * @param threshold Umbral que puede ser positivo (para retrasos) o negativo (para adelantos).
 */
void copyPhase2ThresholdToConstant(int threshold)
{
    cudaMemcpyToSymbol(d_phase2Threshold, &threshold, sizeof(int)); // Copia el umbral a la memoria constante
}

/**
 * @brief Kernel de CUDA para evaluar los retrasos de llegada (ARR_DELAY) en función de un umbral dado.
 * Cada hilo procesa ARR_DELAY y, si cumple el umbral, reserva una posicion de
    salida con atomicAdd y copia retraso y matricula.
 * 
 * @param delayValues  Puntero a los valores de retraso de llegada (ARR_DELAY) en la GPU.
 * @param tailNumIn     Puntero a los números de matrícula (TAIL_NUM) correspondientes a cada registro en la GPU.
 * @param totalElements Número total de elementos en los arreglos de entrada.
 * @param outCount      Puntero al contador de elementos en la salida.
 * @param outDelayValues Puntero a los valores de retraso de llegada en la salida.
 * @param outTailNumBuffer Puntero al buffer de números de matrícula en la salida.
 * @return __global__ 
 */
__global__ void phase2ArrivalDelayKernel(
    const float* delayValues,
    const char* tailNumIn,
    int totalElements,
    int* outCount,
    int* outDelayValues,
    char* outTailNumBuffer)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalElements) {
        return;
    }

    const float rawValue = delayValues[idx]; // Obtenemos el valor de retraso de llegada para este hilo

    if (rawValue != rawValue) {
        return;
    }

    const int delayValue = static_cast<int>(rawValue); // Convertimos el valor de retraso a entero para compararlo con el umbral, asumiendo que el umbral también es un entero

    // comprobamos umbral
    if (!matchesSignedThreshold(delayValue, d_phase2Threshold)) {
        return;
    }

    const int outputIndex = atomicAdd(outCount, 1); // Reservamos una posición en la salida incrementando el contador de forma atómica

    outDelayValues[outputIndex] = delayValue; // Copiamos el valor de retraso que cumple la condición a la salida

    const char* inputTailNum = tailNumIn + idx * kPhase2TailNumStride; // Calculamos la posición del número de matrícula de entrada para este hilo
    char* outputTailNum = outTailNumBuffer + outputIndex * kPhase2TailNumStride; // Calculamos la posición del número de matrícula de salida para este hilo

    // Copiamos el número de matrícula de entrada a la salida (asumiendo que es un string de longitud fija definido por kPhase2TailNumStride)
    for (int i = 0; i < kPhase2TailNumStride; ++i) {
        outputTailNum[i] = inputTailNum[i];
    }

    const char* label = d_phase2Threshold >= 0 ? "Retraso (llegada)" : "Adelanto (llegada)";

    printf("- Hilo #%d  Matricula: %s  %s: %d min\n",
        idx,
        outputTailNum,
        label,
        delayValue);
}

/**
 * @brief Imprime un resumen de los resultados de la Fase 02 en la CPU.
 * 
 * @param threshold Umbral utilizado en la Fase 02.
 * @param resultCount Número de resultados encontrados.
 * @param outDelayValues Vector con los valores de retraso de llegada encontrados.
 * @param outTailNumBuffer Vector con los números de matrícula encontrados.
 */
void printPhase2HostSummary(
    int threshold,
    int resultCount,
    const std::vector<int>& outDelayValues,
    const std::vector<char>& outTailNumBuffer)
{
    const char* label = threshold >= 0 ? "Retraso" : "Adelanto";

    std::cout << "\nResultados CPU:\n";
    std::cout << "Se han encontrado " << resultCount << " aviones\n";

    // Imprimimos los resultados encontrados, mostrando la matrícula y el valor de retraso o adelanto para cada uno
    for (int i = 0; i < resultCount; ++i) {
        const char* tailNum = &outTailNumBuffer[static_cast<std::size_t>(i) * kPhase2TailNumStride];
        const int detectedValue = outDelayValues[static_cast<std::size_t>(i)];

        std::cout << "- Matricula " << tailNum
                  << " | " << label
                  << ": " << detectedValue << " minutos\n";
    }
}

} // namespace

/**
 * @brief Función principal para ejecutar la Fase 02 del análisis de datos de vuelos.
 * Esta función configura el lanzamiento del kernel, ejecuta el kernel para evaluar los retrasos de llegada en función del umbral dado, y luego recopila e imprime los resultados.
 * 
 * @param threshold Umbral que puede ser positivo (para retrasos) o negativo (para adelantos).
 */
void phase02(int threshold)
{
    const LaunchConfig launchConfig = computeLaunchConfig(g_rowCount); // Calculamos la configuración de lanzamiento en función del número total de filas del dataset

    std::cout << "ARR_DELAY + TAIL_NUM | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    cudaMemset(d_phase2Count, 0, sizeof(int)); // Inicializamos el contador de resultados en la GPU a 0 antes de ejecutar el kernel
    copyPhase2ThresholdToConstant(threshold); // Copiamos el umbral a la memoria constante de la GPU para que esté disponible para todos los hilos durante la ejecución del kernel

    // Lanzamos kernel 
    phase2ArrivalDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        d_arrDelay,
        d_tailNums,
        g_rowCount,
        d_phase2Count,
        d_phase2OutDelayValues,
        d_phase2OutTailNums);

        // a esperar al kernel y ver si todo fue bien. No tardara mucho :)
    if (!executeAndWait("phase2ArrivalDelayKernel")) {
        std::cout << "La Fase 02 no se ha podido completar.\n";
        return;
    }

    int resultCount = 0;
    cudaMemcpy(&resultCount, d_phase2Count, sizeof(int), cudaMemcpyDeviceToHost); // Copiamos el número de resultados encontrados desde la GPU al host para saber cuántos resultados hay que copiar y mostrar

    std::vector<int> outDelayValues(static_cast<std::size_t>(resultCount));
    std::vector<char> outTailNumBuffer(static_cast<std::size_t>(resultCount) * kPhase2TailNumStride, '\0');

    // si se han encontrado resultados, los copiamos desde la GPU al host para poder mostrarlos en el resumen de la CPU. Solo copiamos la cantidad de resultados que se han encontrado para evitar copiar memoria innecesaria.
    if (resultCount > 0) {
        cudaMemcpy(
            outDelayValues.data(),
            d_phase2OutDelayValues,
            static_cast<std::size_t>(resultCount) * sizeof(int), // tamaño real de los datos a copiar basado en el número de resultados encontrados
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            outTailNumBuffer.data(),
            d_phase2OutTailNums,
            static_cast<std::size_t>(resultCount) * kPhase2TailNumStride * sizeof(char), // tamaño real de los datos a copiar basado en el número de resultados encontrados y el tamaño de cada matrícula
            cudaMemcpyDeviceToHost);
    }

    printPhase2HostSummary(threshold, resultCount, outDelayValues, outTailNumBuffer); // imprimimos resumen de resultados en host
}
