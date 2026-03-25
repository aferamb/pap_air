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
    deviceCompareReduction

    Comparador minimo comun para las cuatro variantes de la Fase 03. Se deja
    en device para poder reutilizar la misma logica dentro de varios kernels
    sin depender de funciones auxiliares de otras librerias.
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

    Helper interno usado por la variante intermedia cuando el segundo elemento
    de la pareja cae en otro bloque. Calcula, leyendo memoria global, el mejor
    valor entre:

    - la posicion anterior;
    - la posicion actual;
    - la posicion posterior.

    Si alguna posicion no existe, se sustituye por la identidad de la
    reduccion para no contaminar el resultado.
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

/*
    reductionBasic

    Variante 3.2. Cada hilo trabaja con una ventana de tres posiciones:

    - anterior;
    - actual;
    - siguiente.

    La informacion vecina se apoya en memoria compartida para que el acceso
    quede claramente visible y separado de la memoria global.
*/
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

    // Cada hilo deja su valor en la zona central de la ventana compartida.
    if (globalIndex < n) {
        sharedWindow[localIndex + 1] = data[globalIndex];
    } else {
        sharedWindow[localIndex + 1] = identity;
    }

    // Un unico hilo carga los halos de izquierda y derecha del bloque.
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

    const int previousValue = sharedWindow[localIndex];
    const int currentValue = sharedWindow[localIndex + 1];
    const int nextValue = sharedWindow[localIndex + 2];

    int bestValue = deviceCompareReduction(previousValue, currentValue, isMax);
    bestValue = deviceCompareReduction(bestValue, nextValue, isMax);

    if (isMax) {
        atomicMax(result, bestValue);
    } else {
        atomicMin(result, bestValue);
    }
}

/*
    reductionIntermediate

    Variante 3.3. La primera parte es igual que la basica: cada hilo consulta
    una ventana de tres posiciones usando memoria compartida. La diferencia es
    que el mejor valor local se guarda en una segunda zona compartida y solo
    los hilos con indice global par publican por parejas hacia memoria global.
*/
__global__ void reductionIntermediate(const int* data, int* result, int n, bool isMax)
{
    extern __shared__ int sharedMemory[];

    // Repartimos la memoria compartida en dos zonas contiguas:
    // una para la ventana con halo y otra para el mejor valor local.
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

    if (globalIndex >= n) {
        return;
    }

    if ((globalIndex % 2) != 0) {
        return;
    }

    int pairBest = sharedLocalBest[localIndex];
    const int nextGlobalIndex = globalIndex + 1;

    if (nextGlobalIndex < n) {
        // Si el siguiente hilo esta dentro del mismo bloque, reutilizamos la
        // memoria compartida. Si no, recalculamos su ventana desde memoria
        // global para no perder la pareja que cruza el limite del bloque.
        if (localIndex + 1 < validElementsInBlock) {
            pairBest = deviceCompareReduction(pairBest, sharedLocalBest[localIndex + 1], isMax);
        } else {
            const int nextBest = computeWindowReductionFromGlobal(data, n, nextGlobalIndex, isMax);
            pairBest = deviceCompareReduction(pairBest, nextBest, isMax);
        }
    }

    if (isMax) {
        atomicMax(result, pairBest);
    } else {
        atomicMin(result, pairBest);
    }
}

/*
    reductionPattern

    Variante 3.4. Cada bloque reduce su tramo del vector a un unico parcial.
    El host relanzara el mismo patron sobre el vector de parciales hasta dejar
    10 elementos o menos, momento en el que cerrara el resultado en CPU.
*/
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

    // Reducimos por parejas sucesivas hasta dejar un unico valor por bloque.
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

/*
    phase4SharedHistogramKernel

    Cada bloque mantiene una copia privada del histograma en memoria
    compartida. Esto reduce la contencion respecto a hacer todas las atomicas
    directamente sobre memoria global.
*/
__global__ void phase4SharedHistogramKernel(
    const int* denseIndices,
    int totalElements,
    int totalBins,
    unsigned int* partialHistograms)
{
    extern __shared__ unsigned int sharedHistogram[];

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;

    // Como puede haber mas bins que hilos por bloque, inicializamos la zona
    // compartida por tramos.
    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        sharedHistogram[bin] = 0;
    }

    __syncthreads();

    if (globalIndex < totalElements) {
        const int denseIndex = denseIndices[globalIndex];
        atomicAdd(&sharedHistogram[denseIndex], 1U);
    }

    __syncthreads();

    // Volcamos el histograma privado del bloque a la matriz global de
    // parciales. Cada bloque escribe una fila completa de totalBins columnas.
    unsigned int* blockPartialHistogram = partialHistograms + blockIdx.x * totalBins;

    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        blockPartialHistogram[bin] = sharedHistogram[bin];
    }
}

/*
    phase4MergeHistogramKernel

    Cada hilo fusiona un bin del histograma final sumando el mismo bin de todos
    los bloques parciales generados en la primera pasada.
*/
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

/*
    phase4GlobalHistogramKernel

    Variante de respaldo cuando el histograma compartido no cabe en memoria
    compartida. Es mas simple, pero genera mas contencion en memoria global.
*/
__global__ void phase4GlobalHistogramKernel(
    const int* denseIndices,
    int totalElements,
    unsigned int* finalHistogram)
{
    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= totalElements) {
        return;
    }

    const int denseIndex = denseIndices[globalIndex];
    atomicAdd(&finalHistogram[denseIndex], 1U);
}
