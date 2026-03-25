#include "kernels.cuh"

#include <limits.h>
#include <stdio.h>

/*
    kernels.cu

    Este archivo contiene la implementacion device/host asociada a los kernels
    CUDA del proyecto. La idea es mantener aqui la logica que realmente se
    ejecuta en GPU para no mezclarla con la parte de menu, carga de CSV o
    preparacion de buffers que vive en main.cu.
*/

// Configuracion minima de la Fase 02 almacenada en memoria constante. El
// enunciado exige usar este tipo de memoria al menos para guardar el criterio
// de filtrado, y aqui guardamos tanto el umbral como el modo.
__constant__ int d_phase2Mode;
__constant__ int d_phase2Threshold;

/*
    matchesDelayFilterMode

    Evalua el criterio comun de las Fases 01 y 02 usando:

    - mode = 1 -> retraso: valor >= threshold
    - mode = 2 -> adelanto: valor <= -threshold
    - mode = 3 -> ambos: cualquiera de las dos condiciones
*/
__device__ bool matchesDelayFilterMode(int value, int mode, int threshold)
{
    if (mode == 1) {
        return value >= threshold;
    }

    if (mode == 2) {
        return value <= -threshold;
    }

    return value >= threshold || value <= -threshold;
}

/*
    detectDeviceDelayLabel

    Devuelve la etiqueta a imprimir para un valor ya detectado. En modo
    "ambos", la etiqueta depende del propio valor concreto.
*/
__device__ const char* detectDeviceDelayLabel(int value, int mode, int threshold)
{
    if (mode == 1) {
        return "Retraso";
    }

    if (mode == 2) {
        return "Adelanto";
    }

    if (value >= threshold) {
        return "Retraso";
    }

    return "Adelanto";
}

/*
    detectDeviceArrivalLabel

    Variante textual especifica de la Fase 02 para que la salida desde GPU use
    exactamente la etiqueta de llegada y no una deduccion indirecta.
*/
__device__ const char* detectDeviceArrivalLabel(int value, int mode, int threshold)
{
    if (mode == 1) {
        return "Retraso (llegada)";
    }

    if (mode == 2) {
        return "Adelanto (llegada)";
    }

    if (value >= threshold) {
        return "Retraso (llegada)";
    }

    return "Adelanto (llegada)";
}

/*
    copyPhase2FilterConfigToConstant

    Copia desde host a memoria constante la configuracion que necesita la Fase
    02: modo y umbral absoluto.
*/
cudaError_t copyPhase2FilterConfigToConstant(int mode, int threshold)
{
    cudaError_t status = cudaMemcpyToSymbol(d_phase2Mode, &mode, sizeof(int));

    if (status != cudaSuccess) {
        return status;
    }

    return cudaMemcpyToSymbol(d_phase2Threshold, &threshold, sizeof(int));
}

/*
    phase1DepartureDelayKernel

    Cada hilo procesa una fila del vector DEP_DELAY ya truncado a entero. La
    mascara de validez permite ignorar datos que en host eran NAN sin tener que
    compactar el dataset ni perder el indice global original.
*/
__global__ void phase1DepartureDelayKernel(
    const int* delayValues,
    const unsigned char* validMask,
    int totalElements,
    int mode,
    int threshold)
{
    // Formula clasica de acceso linealizado 1D en CUDA.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Si sobran hilos respecto al tamano real del vector, salen sin tocar
    // memoria para evitar accesos fuera de rango.
    if (idx >= totalElements) {
        return;
    }

    // Si el dato original era NAN, el host deja una mascara a 0 para que este
    // hilo ignore la posicion y no imprima un valor inventado.
    if (validMask[idx] == 0) {
        return;
    }

    const int delayValue = delayValues[idx];

    const bool matchesThreshold = matchesDelayFilterMode(delayValue, mode, threshold);

    if (matchesThreshold) {
        const char* detectedLabel = detectDeviceDelayLabel(delayValue, mode, threshold);
        printf("- Hilo #%d: %s de %d minutos\n", idx, detectedLabel, delayValue);
    }
}

/*
    phase2ArrivalDelayKernel

    Cada hilo analiza una fila de ARR_DELAY y, si cumple el umbral constante,
    reserva una posicion libre en la salida con una operacion atomica. Despues
    copia tanto el retraso detectado como la matricula asociada.
*/
__global__ void phase2ArrivalDelayKernel(
    const int* delayValues,
    const unsigned char* validMask,
    const char* tailNumIn,
    int totalElements,
    int* outCount,
    int* outDelayValues,
    char* outTailNumBuffer)
{
    // Cada hilo trabaja sobre una unica fila del dataset.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalElements) {
        return;
    }

    if (validMask[idx] == 0) {
        return;
    }

    const int delayValue = delayValues[idx];

    // El modo y el umbral viven en memoria constante para cumplir el requisito
    // del enunciado y porque son valores comunes a todos los hilos.
    const bool matchesThreshold =
        matchesDelayFilterMode(delayValue, d_phase2Mode, d_phase2Threshold);

    if (!matchesThreshold) {
        return;
    }

    // atomicAdd devuelve el hueco reservado para este hilo dentro de la salida.
    const int outputIndex = atomicAdd(outCount, 1);

    outDelayValues[outputIndex] = delayValue;

    // Calculamos el inicio de la matricula de entrada y el de la salida
    // usando un stride fijo para mantener un layout muy simple.
    const char* inputTailNum = tailNumIn + idx * kPhase2TailNumStride;
    char* outputTailNum = outTailNumBuffer + outputIndex * kPhase2TailNumStride;

    // Copiamos toda la celda fija. El host ya garantiza que la cadena esta
    // terminada en '\0', asi que la salida seguira siendo imprimible.
    for (int i = 0; i < kPhase2TailNumStride; ++i) {
        outputTailNum[i] = inputTailNum[i];
    }

    // La salida puede intercalarse entre hilos. Eso es normal con printf
    // desde GPU y no afecta al resultado almacenado en memoria.
    const char* detectedLabel =
        detectDeviceArrivalLabel(delayValue, d_phase2Mode, d_phase2Threshold);

    printf("- Hilo #%d  Matricula: %s  %s: %d min\n",
        idx,
        outputTailNum,
        detectedLabel,
        delayValue);
}

/*
    reductionSimple

    Kernel de reduccion simple que ya existia en el repositorio. Se conserva
    como base de la variante 3.1 de la Fase 03.
*/
__global__ void reductionSimple(int* data, int* result, int n, bool isMax)
{
    // Formula clasica de acceso linealizado 1D en CUDA.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Si sobran hilos respecto al tamano real del vector, esos hilos salen
    // inmediatamente para no leer fuera de rango.
    if (idx >= n) {
        return;
    }

    // Cada hilo solo lee el elemento que le corresponde dentro del vector.
    const int value = data[idx];

    // La rama selecciona el tipo de reduccion pedido por el host. Ambas usan
    // operaciones atomicas porque varios hilos pueden intentar actualizar el
    // mismo acumulador global al mismo tiempo.
    if (isMax) {
        atomicMax(result, value);
    } else {
        atomicMin(result, value);
    }
}
