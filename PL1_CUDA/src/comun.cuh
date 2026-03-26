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

// Tamano fijo de la matricula linealizada para la Fase 02. (para el bucle)
constexpr int kPhase2TailNumStride = 16;

// Estructura para almacenar la configuración de lanzamiento de los kernels CUDA.
struct LaunchConfig {
    int blocks = 0;
    int threadsPerBlock = 1;
};

/*
    Estado global del programa.

    Se declara como extern para que exista una unica definicion real en
    comun.cu y el resto de modulos pueda reutilizarlo.
*/
extern DatasetColumns g_dataset;                       // Estructura que contiene las columnas del dataset cargado
extern LoadSummary g_summary;                          // Estructura que contiene el resumen de la carga del dataset
extern std::string g_datasetPath;                      // Ruta del dataset cargado, se mantiene para mostrarla en el resumen de la GPU
extern bool g_datasetLoaded;                            // Indica si el dataset se ha cargado correctamente, se utiliza para mostrar información relevante en el resumen de la GPU

extern bool g_deviceReady;                          // Indica si la GPU está lista para ejecutar kernels, se establece después de consultar la información de la GPU    
extern cudaDeviceProp g_deviceProp;                 // Estructura que contiene las propiedades de la GPU, se llena al consultar la información de la GPU
extern std::string g_deviceErrorMessage;            // Mensaje de error relacionado con la GPU, se establece si ocurre un error al consultar la información de la GPU

extern int g_rowCount;                              // Número total de filas del dataset, se mantiene para mostrarlo en el resumen de la GPU y para calcular la configuración de lanzamiento base

extern float* d_depDelay;                           // Puntero a la columna de retrasos de salida en la GPU
extern float* d_arrDelay;                           // Puntero a la columna de retrasos de llegada en la GPU
extern char* d_tailNums;                            // Puntero a la columna de tailNums en la GPU, se mantiene como char* para facilitar el manejo de strings de longitud variable
// TODO:extra, es necesario?
extern int* d_originSeqId;                          // Puntero a la columna de originSeqId en la GPU
extern int* d_destSeqId;                            // Puntero a la columna de destSeqId en la GPU
extern int* d_originAirportCodes;                   // Puntero a la columna de códigos de aeropuerto de origen en la GPU, se mantiene como int* para facilitar el manejo de códigos de aeropuerto mapeados a enteros
extern int* d_destAirportCodes;                     // Puntero a la columna de códigos de aeropuerto de destino en la GPU, se mantiene como int* para facilitar el manejo de códigos de aeropuerto mapeados a enteros

extern int* d_phase2Count;                          // Puntero a la variable que almacena el conteo de filas que cumplen la condición en la Fase 02, se mantiene como int* para facilitar su actualización desde el kernel
extern int* d_phase2OutDelayValues;                 // Puntero a la columna de valores de retraso de salida que cumplen la condición en la Fase 02, se mantiene como float* para facilitar su manejo como valores numéricos
extern char* d_phase2OutTailNums;                   // Puntero a la columna de tailNums que cumplen la condición en la Fase 02, se mantiene como char* para facilitar el manejo de strings de longitud variable

extern int* d_originDenseInput;                     // Puntero a la columna de originSeqId mapeada a IDs densos en la GPU, se mantiene como int* para facilitar el manejo de IDs densos
extern int g_originTotalElements;                  // Número total de elementos en la columna de originSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
extern int g_originTotalBins;                      // Número total de bins únicos en la columna de originSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
extern std::vector<int> g_originDenseToSeqId;       // Vector que mapea los IDs densos de originSeqId a los IDs originales

extern int* d_destinationDenseInput;         // Puntero a la columna de destSeqId mapeada a IDs densos en la GPU, se mantiene como int* para facilitar el manejo de IDs densos       
extern int g_destinationTotalElements;             // Número total de elementos en la columna de destSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
extern int g_destinationTotalBins;                 // Número total de bins únicos en la columna de destSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
extern std::vector<int> g_destinationDenseToSeqId;     // Vector que mapea los IDs densos de destSeqId a los IDs originales

bool executeAndWait(const char* context);
bool queryGpuInfo();
LaunchConfig computeLaunchConfig(int totalElements);
void printGpuSummary();
