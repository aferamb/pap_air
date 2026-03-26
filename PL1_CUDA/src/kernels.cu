#include "kernels.cuh"

#include <limits.h>
#include <stdio.h>

/*
    kernels.cu

    Este archivo contiene la logica que se ejecuta en GPU. La parte de menu,
    carga del CSV y preparacion del estado vive en main.cu.
*/

// La Fase 02 mantiene en memoria constante solo el umbral firmado, que es la
// configuracion comun a todos los hilos.
__constant__ int d_phase2Threshold;

/*
    matchesSignedThreshold

    Regla comun para Fase 01 y Fase 02:

    - threshold >= 0 -> buscar retrasos: value >= threshold
    - threshold < 0 -> buscar adelantos: value <= threshold
*/
__device__ bool matchesSignedThreshold(int value, int threshold)
{
    if (threshold >= 0) {
        return value >= threshold;
    }

    return value <= threshold;
}

/*
    deviceCompareReduction

    Comparador minimo comun para las cuatro variantes de la Fase 03.
*/
__device__ int deviceCompareReduction(int left, int right, bool isMax)
{
    if (isMax) {
        return left > right ? left : right;
    }

    return left < right ? left : right;
}

/*
    computeWindowReductionFromGlobal

    Helper interno usado por la variante intermedia cuando la pareja cruza el
    limite de bloque.
*/
__device__ int computeWindowReductionFromGlobal(const int* data, int n, int idx, bool isMax)
{
    const int identity = isMax ? INT_MIN : INT_MAX;
    int bestValue = data[idx];

    if (idx > 0) {
        bestValue = deviceCompareReduction(bestValue, data[idx - 1], isMax);
    } else {
        bestValue = deviceCompareReduction(bestValue, identity, isMax);
    }

    if (idx + 1 < n) {
        bestValue = deviceCompareReduction(bestValue, data[idx + 1], isMax);
    } else {
        bestValue = deviceCompareReduction(bestValue, identity, isMax);
    }

    return bestValue;
}

cudaError_t copyPhase2ThresholdToConstant(int threshold)
{
    return cudaMemcpyToSymbol(d_phase2Threshold, &threshold, sizeof(int));
}

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

__global__ void reductionSimple(int* data, int* result, int n, bool isMax)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const int value = data[idx];

    if (isMax) {
        atomicMax(result, value);
    } else {
        atomicMin(result, value);
    }
}

__global__ void reductionBasic(const int* data, int* result, int n, bool isMax)
{
    extern __shared__ int sharedWindow[];

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int identity = isMax ? INT_MIN : INT_MAX;

    int validElementsInBlock = n - blockStart;

    if (validElementsInBlock < 0) {
        validElementsInBlock = 0;
    }

    if (validElementsInBlock > blockDim.x) {
        validElementsInBlock = blockDim.x;
    }

    if (globalIndex < n) {
        sharedWindow[localIndex + 1] = data[globalIndex];
    } else {
        sharedWindow[localIndex + 1] = identity;
    }

    if (localIndex == 0) {
        sharedWindow[0] = blockStart > 0 ? data[blockStart - 1] : identity;

        if (validElementsInBlock > 0) {
            const int nextIndex = blockStart + validElementsInBlock;
            sharedWindow[validElementsInBlock + 1] = nextIndex < n ? data[nextIndex] : identity;
        } else {
            sharedWindow[1] = identity;
        }
    }

    __syncthreads();

    if (globalIndex >= n) {
        return;
    }

    int bestValue = deviceCompareReduction(sharedWindow[localIndex], sharedWindow[localIndex + 1], isMax);
    bestValue = deviceCompareReduction(bestValue, sharedWindow[localIndex + 2], isMax);

    if (isMax) {
        atomicMax(result, bestValue);
    } else {
        atomicMin(result, bestValue);
    }
}

__global__ void reductionIntermediate(const int* data, int* result, int n, bool isMax)
{
    extern __shared__ int sharedMemory[];

    int* sharedWindow = sharedMemory;
    int* sharedLocalBest = sharedMemory + blockDim.x + 2;

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int identity = isMax ? INT_MIN : INT_MAX;

    int validElementsInBlock = n - blockStart;

    if (validElementsInBlock < 0) {
        validElementsInBlock = 0;
    }

    if (validElementsInBlock > blockDim.x) {
        validElementsInBlock = blockDim.x;
    }

    if (globalIndex < n) {
        sharedWindow[localIndex + 1] = data[globalIndex];
    } else {
        sharedWindow[localIndex + 1] = identity;
    }

    if (localIndex == 0) {
        sharedWindow[0] = blockStart > 0 ? data[blockStart - 1] : identity;

        if (validElementsInBlock > 0) {
            const int nextIndex = blockStart + validElementsInBlock;
            sharedWindow[validElementsInBlock + 1] = nextIndex < n ? data[nextIndex] : identity;
        } else {
            sharedWindow[1] = identity;
        }
    }

    __syncthreads();

    if (globalIndex < n) {
        int bestValue = deviceCompareReduction(sharedWindow[localIndex], sharedWindow[localIndex + 1], isMax);
        bestValue = deviceCompareReduction(bestValue, sharedWindow[localIndex + 2], isMax);
        sharedLocalBest[localIndex] = bestValue;
    } else {
        sharedLocalBest[localIndex] = identity;
    }

    __syncthreads();

    if (globalIndex >= n || (globalIndex % 2) != 0) {
        return;
    }

    int pairBest = sharedLocalBest[localIndex];
    const int nextGlobalIndex = globalIndex + 1;

    if (nextGlobalIndex < n) {
        if (localIndex + 1 < validElementsInBlock) {
            pairBest = deviceCompareReduction(pairBest, sharedLocalBest[localIndex + 1], isMax);
        } else {
            pairBest = deviceCompareReduction(pairBest, computeWindowReductionFromGlobal(data, n, nextGlobalIndex, isMax), isMax);
        }
    }

    if (isMax) {
        atomicMax(result, pairBest);
    } else {
        atomicMin(result, pairBest);
    }
}

__global__ void reductionPattern(const int* input, int* output, int n, bool isMax)
{
    extern __shared__ int sharedReduction[];

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;
    const int identity = isMax ? INT_MIN : INT_MAX;

    if (globalIndex < n) {
        sharedReduction[localIndex] = input[globalIndex];
    } else {
        sharedReduction[localIndex] = identity;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localIndex < stride) {
            sharedReduction[localIndex] =
                deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + stride], isMax);
        }

        __syncthreads();
    }

    if (localIndex == 0) {
        output[blockIdx.x] = sharedReduction[0];
    }
}

__global__ void phase4SharedHistogramKernel(
    const int* denseIndices,
    int totalElements,
    int totalBins,
    unsigned int* partialHistograms)
{
    extern __shared__ unsigned int sharedHistogram[];

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;

    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        sharedHistogram[bin] = 0;
    }

    __syncthreads();

    if (globalIndex < totalElements) {
        atomicAdd(&sharedHistogram[denseIndices[globalIndex]], 1U);
    }

    __syncthreads();

    unsigned int* blockPartialHistogram = partialHistograms + blockIdx.x * totalBins;

    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        blockPartialHistogram[bin] = sharedHistogram[bin];
    }
}

__global__ void phase4MergeHistogramKernel(
    const unsigned int* partialHistograms,
    int partialCount,
    int totalBins,
    unsigned int* finalHistogram)
{
    const int binIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (binIndex >= totalBins) {
        return;
    }

    unsigned int totalCount = 0;

    for (int partialIndex = 0; partialIndex < partialCount; ++partialIndex) {
        totalCount += partialHistograms[partialIndex * totalBins + binIndex];
    }

    finalHistogram[binIndex] = totalCount;
}
