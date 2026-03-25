#pragma once

#include <cuda_runtime.h>

/*
    kernels.cuh

    Este header reune los kernels CUDA que ya forman parte del proyecto.

    Estado actual:

    - Fase 01: deteccion de retrasos o adelantos en despegues sobre DEP_DELAY;
    - Fase 02: deteccion de retrasos o adelantos en aterrizajes sobre
      ARR_DELAY + TAIL_NUM usando memoria constante y atomicas;
    - Fase 03: las cuatro variantes de reduccion pedidas en el enunciado.
    - Fase 04: histograma de aeropuertos usando SEQ_ID densos.

    La idea es que main.cu pueda incluir este archivo y disponer de las
    declaraciones necesarias sin mezclar la implementacion CUDA dentro del
    orquestador principal.
*/

// Tamano fijo reservado para cada matricula en la Fase 02. Se usa un stride
// simple para poder linealizar strings en un unico buffer de chars.
constexpr int kPhase2TailNumStride = 16;

/*
    phase1DepartureDelayKernel

    Kernel de la Fase 01. Cada hilo analiza una posicion del vector DEP_DELAY
    ya truncado a entero en host. Si el dato es valido y supera el criterio del
    umbral, el hilo muestra por consola:

    - su identificador global;
    - el valor detectado.

    Parametros:

    - delayValues: vector de retrasos truncados a entero;
    - validMask: mascara 0/1 para ignorar posiciones que en host eran NAN;
    - totalElements: numero total de filas procesables;
    - mode: 1=retraso, 2=adelanto, 3=ambos;
    - threshold: umbral absoluto no negativo introducido por el usuario.
*/
__global__ void phase1DepartureDelayKernel(
    const int* delayValues,
    const unsigned char* validMask,
    int totalElements,
    int mode,
    int threshold);

/*
    copyPhase2FilterConfigToConstant

    Helper host para copiar a memoria constante la configuracion minima de la
    Fase 02: modo y umbral.
*/
cudaError_t copyPhase2FilterConfigToConstant(int mode, int threshold);

/*
    phase2ArrivalDelayKernel

    Kernel de la Fase 02. Cada hilo analiza una posicion de ARR_DELAY y, si la
    posicion cumple el criterio del modo y del umbral almacenados en memoria
    constante:

    - reserva una salida con atomicAdd;
    - guarda retraso y matricula en arrays simples de salida;
    - muestra la informacion por consola desde GPU.

    Parametros:

    - delayValues: vector ARR_DELAY truncado a entero;
    - validMask: mascara 0/1 para ignorar NAN;
    - tailNumIn: buffer linealizado de matriculas de entrada;
    - totalElements: numero total de filas;
    - outCount: contador global de resultados encontrados;
    - outDelayValues: array simple de retrasos detectados;
    - outTailNumBuffer: array simple de matriculas detectadas.
*/
__global__ void phase2ArrivalDelayKernel(
    const int* delayValues,
    const unsigned char* validMask,
    const char* tailNumIn,
    int totalElements,
    int* outCount,
    int* outDelayValues,
    char* outTailNumBuffer);

/*
    deviceCompareReduction

    Helper device muy pequeno para comparar dos enteros sin depender de
    funciones max()/min() externas. Devuelve:

    - el mayor, si isMax es true;
    - el menor, si isMax es false.
*/
__device__ int deviceCompareReduction(int left, int right, bool isMax);

/*
    reductionSimple

    Kernel de reduccion simple asociado a la variante 3.1 de la Fase 03.

    Parametros:

    - data: vector de entrada en memoria global de GPU;
    - result: puntero a un unico acumulador global en device;
    - n: numero total de elementos del vector;
    - isMax: selector de operacion, true para maximo y false para minimo.

    Cada hilo:

    1. calcula su indice global 1D;
    2. comprueba si ese indice esta dentro del rango;
    3. lee su elemento de entrada;
    4. aplica una operacion atomica sobre el acumulador global.

    Es un enfoque didactico y funcional, pero no el mas eficiente, porque todos
    los hilos compiten por la misma posicion de memoria global.
*/
__global__ void reductionSimple(int* data, int* result, int n, bool isMax);

/*
    reductionBasic

    Variante 3.2 de la Fase 03. Cada hilo:

    - consulta anterior, actual y siguiente;
    - usa memoria compartida para esas lecturas vecinas;
    - publica el mejor de las tres posiciones en un acumulador global atomico.
*/
__global__ void reductionBasic(const int* data, int* result, int n, bool isMax);

/*
    reductionIntermediate

    Variante 3.3 de la Fase 03. Cada hilo calcula primero, en memoria
    compartida, el mejor valor de su ventana de tres posiciones. Despues, los
    hilos con indice global par comparan su valor local con el siguiente y
    publican el mejor mediante una operacion atomica global.
*/
__global__ void reductionIntermediate(const int* data, int* result, int n, bool isMax);

/*
    reductionPattern

    Variante 3.4 de la Fase 03. Aplica un patron clasico de reduccion por
    bloque usando memoria compartida y genera un vector de resultados
    parciales. El host debera relanzar este kernel tantas veces como haga
    falta hasta dejar 10 elementos o menos.
*/
__global__ void reductionPattern(const int* input, int* output, int n, bool isMax);

/*
    phase4SharedHistogramKernel

    Kernel principal recomendado para la Fase 04 cuando el numero de bins cabe
    en memoria compartida. Cada bloque:

    - inicializa su histograma compartido;
    - procesa sus filas validas;
    - vuelca su copia privada a un buffer global de parciales.
*/
__global__ void phase4SharedHistogramKernel(
    const int* denseIndices,
    int totalElements,
    int totalBins,
    unsigned int* partialHistograms);

/*
    phase4MergeHistogramKernel

    Fusiona el buffer de histogramas parciales generado por
    phase4SharedHistogramKernel y deja un unico histograma final en memoria
    global.
*/
__global__ void phase4MergeHistogramKernel(
    const unsigned int* partialHistograms,
    int partialCount,
    int totalBins,
    unsigned int* finalHistogram);
