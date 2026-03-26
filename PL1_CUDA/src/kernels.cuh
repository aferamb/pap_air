#pragma once

#include <cuda_runtime.h>

/*
    kernels.cuh

    Este header declara los kernels CUDA del proyecto y deja separada la parte
    device del flujo host que vive en main.cu.

    Estado funcional:

    - Fase 01: filtro firmado sobre DEP_DELAY;
    - Fase 02: filtro firmado sobre ARR_DELAY + TAIL_NUM con memoria constante;
    - Fase 03: cuatro variantes de reduccion;
    - Fase 04: histograma compartido por bins densos.
*/

// Tamano fijo de cada matricula linealizada para la Fase 02.
constexpr int kPhase2TailNumStride = 16;

/*
    phase1DepartureDelayKernel

    Cada hilo analiza una fila de DEP_DELAY. El umbral es firmado:

    - threshold >= 0 -> buscar retrasos;
    - threshold < 0 -> buscar adelantos.
*/
__global__ void phase1DepartureDelayKernel(
    const float* delayValues,
    int totalElements,
    int threshold);

/*
    copyPhase2ThresholdToConstant

    Copia a memoria constante el umbral firmado de la Fase 02.
*/
cudaError_t copyPhase2ThresholdToConstant(int threshold);

/*
    phase2ArrivalDelayKernel

    Cada hilo analiza ARR_DELAY y, si cumple el umbral firmado almacenado en
    memoria constante, reserva una posicion de salida con atomicAdd y copia:

    - el retraso truncado a entero;
    - la matricula correspondiente.
*/
__global__ void phase2ArrivalDelayKernel(
    const float* delayValues,
    const char* tailNumIn,
    int totalElements,
    int* outCount,
    int* outDelayValues,
    char* outTailNumBuffer);

/*
    deviceCompareReduction

    Comparador minimo comun para las reducciones de la Fase 03.
*/
__device__ int deviceCompareReduction(int left, int right, bool isMax);

/*
    Kernels de la Fase 03.
*/
__global__ void reductionSimple(int* data, int* result, int n, bool isMax);
__global__ void reductionBasic(const int* data, int* result, int n, bool isMax);
__global__ void reductionIntermediate(const int* data, int* result, int n, bool isMax);
__global__ void reductionPattern(const int* input, int* output, int n, bool isMax);

/*
    Kernels de la Fase 04.
*/
__global__ void phase4SharedHistogramKernel(
    const int* denseIndices,
    int totalElements,
    int totalBins,
    unsigned int* partialHistograms);

__global__ void phase4MergeHistogramKernel(
    const unsigned int* partialHistograms,
    int partialCount,
    int totalBins,
    unsigned int* finalHistogram);
