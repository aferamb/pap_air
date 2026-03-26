#include "parte2.cuh"

#include <stdio.h>

#include <iostream>
#include <vector>

#include "comun.cuh"

namespace {

// Umbral compartido por todos los hilos de la Fase 02.
__constant__ int d_phase2Threshold;

__device__ bool matchesSignedThreshold(int value, int threshold)
{
    if (threshold >= 0) {
        return value >= threshold;
    }

    return value <= threshold;
}

void copyPhase2ThresholdToConstant(int threshold)
{
    cudaMemcpyToSymbol(d_phase2Threshold, &threshold, sizeof(int));
}

/*
    Cada hilo procesa ARR_DELAY y, si cumple el umbral, reserva una posicion de
    salida con atomicAdd y copia retraso y matricula.
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

    const float rawValue = delayValues[idx];

    if (rawValue != rawValue) {
        return;
    }

    const int delayValue = static_cast<int>(rawValue);

    if (!matchesSignedThreshold(delayValue, d_phase2Threshold)) {
        return;
    }

    const int outputIndex = atomicAdd(outCount, 1);

    outDelayValues[outputIndex] = delayValue;

    const char* inputTailNum = tailNumIn + idx * kPhase2TailNumStride;
    char* outputTailNum = outTailNumBuffer + outputIndex * kPhase2TailNumStride;

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

void printPhase2HostSummary(
    int threshold,
    int resultCount,
    const std::vector<int>& outDelayValues,
    const std::vector<char>& outTailNumBuffer)
{
    const char* label = threshold >= 0 ? "Retraso" : "Adelanto";

    std::cout << "\nResultados CPU:\n";
    std::cout << "Se han encontrado " << resultCount << " aviones\n";

    for (int i = 0; i < resultCount; ++i) {
        const char* tailNum = &outTailNumBuffer[static_cast<std::size_t>(i) * kPhase2TailNumStride];
        const int detectedValue = outDelayValues[static_cast<std::size_t>(i)];

        std::cout << "- Matricula " << tailNum
                  << " | " << label
                  << ": " << detectedValue << " minutos\n";
    }
}

} // namespace

void phase02(int threshold)
{
    const LaunchConfig launchConfig = computeLaunchConfig(g_rowCount);

    std::cout << "ARR_DELAY + TAIL_NUM | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    cudaMemset(d_phase2Count, 0, sizeof(int));
    copyPhase2ThresholdToConstant(threshold);

    phase2ArrivalDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        d_arrDelay,
        d_tailNums,
        g_rowCount,
        d_phase2Count,
        d_phase2OutDelayValues,
        d_phase2OutTailNums);

    if (!executeAndWait("phase2ArrivalDelayKernel")) {
        std::cout << "La Fase 02 no se ha podido completar.\n";
        return;
    }

    int resultCount = 0;
    cudaMemcpy(&resultCount, d_phase2Count, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> outDelayValues(static_cast<std::size_t>(resultCount));
    std::vector<char> outTailNumBuffer(static_cast<std::size_t>(resultCount) * kPhase2TailNumStride, '\0');

    if (resultCount > 0) {
        cudaMemcpy(
            outDelayValues.data(),
            d_phase2OutDelayValues,
            static_cast<std::size_t>(resultCount) * sizeof(int),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            outTailNumBuffer.data(),
            d_phase2OutTailNums,
            static_cast<std::size_t>(resultCount) * kPhase2TailNumStride * sizeof(char),
            cudaMemcpyDeviceToHost);
    }

    printPhase2HostSummary(threshold, resultCount, outDelayValues, outTailNumBuffer);
}
