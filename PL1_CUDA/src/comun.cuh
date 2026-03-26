#pragma once

#include <cuda_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "csv_reader.h"

/*
    comun.cuh

    Este modulo declara el estado global compartido y las utilidades CUDA que
    se reutilizan en varias fases. La idea es mantener un unico punto comun y
    evitar pasar structs grandes entre modulos.
*/

// Tamano fijo de la matricula linealizada para la Fase 02.
constexpr int kPhase2TailNumStride = 16;

struct LaunchConfig {
    int blocks = 0;
    int threadsPerBlock = 1;
};

/*
    Estado global del programa.

    Se declara como extern para que exista una unica definicion real en
    comun.cu y el resto de modulos pueda reutilizarlo sin firmas largas.
*/
extern DatasetColumns g_dataset;
extern LoadSummary g_summary;
extern std::string g_datasetPath;
extern bool g_datasetLoaded;

extern bool g_deviceReady;
extern cudaDeviceProp g_deviceProp;
extern std::string g_deviceErrorMessage;

extern int g_rowCount;

extern float* d_depDelay;
extern float* d_arrDelay;
extern char* d_tailNums;

extern int* d_phase2Count;
extern int* d_phase2OutDelayValues;
extern char* d_phase2OutTailNums;

extern int* d_originDenseInput;
extern int g_originTotalElements;
extern int g_originTotalBins;
extern std::vector<int> g_originDenseToSeqId;

extern int* d_destinationDenseInput;
extern int g_destinationTotalElements;
extern int g_destinationTotalBins;
extern std::vector<int> g_destinationDenseToSeqId;

bool executeAndWait(const char* context);
bool queryGpuInfo();
LaunchConfig computeLaunchConfig(int totalElements);
void printGpuSummary();
