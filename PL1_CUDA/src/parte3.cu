#include "parte3.cuh"

#include <climits>
#include <cmath>
#include <iostream>
#include <vector>

#include "comun.cuh"

namespace {

/*
variantes de reducción atómica (Simple, Basica e Intermedia)
*/
enum class Phase3AtomicVariant {
    Simple,
    Basic,
    Intermediate
};

/** @brief Compara dos valores enteros y devuelve el máximo o mínimo según el parámetro isMax.
 * @param left Primer valor a comparar.
 * @param right Segundo valor a comparar.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 * @return El valor máximo o mínimo según el parámetro isMax.
 */
__device__ int deviceCompareReduction(int left, int right, bool isMax)
{
    if (isMax) {
        return left > right ? left : right; // Maximo
    }

    return left < right ? left : right; // Minimo
}

/** @brief Calcula la reducción de una ventana desde la memoria global.
 * @param data Puntero a los datos.
 * @param n Número de elementos.
 * @param idx Índice del elemento actual.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 * @return El valor máximo o mínimo de la ventana.
 */
__device__ int computeWindowReductionFromGlobal(const int* data, int n, int idx, bool isMax)
{
    const int identity = isMax ? INT_MIN : INT_MAX;
    int bestValue = data[idx];

    if (idx > 0) {
        bestValue = deviceCompareReduction(bestValue, data[idx - 1], isMax); // Comparar con el elemento anterior
    } else {
        bestValue = deviceCompareReduction(bestValue, identity, isMax); // Si no hay elemento anterior, comparar con el valor identidad
    }

    if (idx + 1 < n) {
        bestValue = deviceCompareReduction(bestValue, data[idx + 1], isMax); // Comparar con el elemento siguiente
    } else {
        bestValue = deviceCompareReduction(bestValue, identity, isMax); // Si no hay elemento siguiente, comparar con el valor identidad
    }

    return bestValue;
}

/** @brief Realiza una reducción simple en la memoria global.
 * @param data Puntero a los datos.
 * @param result Puntero al resultado.
 * @param n Número de elementos.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 */
__global__ void reductionSimple(int* data, int* result, int n, bool isMax)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int value = data[idx];
    if (isMax) {
        atomicMax(result, value); // Reducción atómica para máximo
    } else {
        atomicMin(result, value); // Reducción atómica para mínimo
    }
}

/** @brief Realiza una reducción básica en la memoria compartida.
 *  Esta reduccion la realiza cada hilo comparando su elemento con el anterior y el siguiente, y luego realiza una reducción atómica con el mejor valor encontrado. 
 * y guarda el resultado en memoria global. 
 * @param data Puntero a los datos.
 * @param result Puntero al resultado.
 * @param n Número de elementos.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 */
__global__ void reductionBasic(const int* data, int* result, int n, bool isMax)
{
    extern __shared__ int sharedWindow[]; // Tamaño de bloque + 2 para almacenar el elemento anterior y siguiente
    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; // Índice global del hilo
    const int localIndex = threadIdx.x; // Índice local dentro del bloque
    const int blockStart = blockIdx.x * blockDim.x; // Índice del primer elemento del bloque
    const int identity = isMax ? INT_MIN : INT_MAX; 
    int validElementsInBlock = n - blockStart; // Número de elementos válidos en el bloque
    // Ajustar el número de elementos válidos en el bloque para no exceder el tamaño del bloque
    if (validElementsInBlock < 0) {
        validElementsInBlock = 0;
    }
    if (validElementsInBlock > blockDim.x) {
        validElementsInBlock = blockDim.x;
    }
    sharedWindow[localIndex + 1] = globalIndex < n ? data[globalIndex] : identity; /// Cargar el elemento actual en la ventana compartida, o el valor identidad si el índice global excede el número de elementos
    if (localIndex == 0) {
        sharedWindow[0] = blockStart > 0 ? data[blockStart - 1] : identity; // Cargar el elemento anterior al bloque en la ventana compartida, o el valor identidad si no hay elemento anterior
        if (validElementsInBlock > 0) {
            const int nextIndex = blockStart + validElementsInBlock; // Índice del elemento siguiente al bloque
            sharedWindow[validElementsInBlock + 1] = nextIndex < n ? data[nextIndex] : identity; // Cargar el elemento siguiente al bloque en la ventana compartida, o el valor identidad si no hay elemento siguiente
        } else {
            sharedWindow[1] = identity; // Si no hay elementos válidos en el bloque, cargar el valor identidad en la posición del siguiente elemento
        }
    }

    __syncthreads(); // Sincronizar para asegurarse de que todos los hilos han cargado sus datos en la ventana compartida

    if (globalIndex >= n) {
        return;
    }
    int bestValue = deviceCompareReduction(sharedWindow[localIndex], sharedWindow[localIndex + 1], isMax); // Comparar el elemento actual con el siguiente en la ventana compartida
    bestValue = deviceCompareReduction(bestValue, sharedWindow[localIndex + 2], isMax); // Comparar el mejor valor encontrado con el elemento siguiente al siguiente en la ventana compartida
    if (isMax) {
        atomicMax(result, bestValue); // Reducción atómica para máximo con el mejor valor encontrado
    } else {
        atomicMin(result, bestValue); // Reducción atómica para mínimo con el mejor valor encontrado
    }
}

/** @brief Realiza una reducción intermedia en la memoria compartida.
 * Esta reducción es similar a la reducción básica, pero además de comparar el elemento actual con el anterior y el siguiente, 
 * también compara el mejor valor encontrado por cada hilo con el mejor valor encontrado por su hilo vecino (hilo par con hilo impar) 
 * para obtener un resultado antes de realizar la reducción atómica en memoria global.
 * @param data Puntero a los datos.
 * @param result Puntero al resultado.
 * @param n Número de elementos.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 */
__global__ void reductionIntermediate(const int* data, int* result, int n, bool isMax)
{
    extern __shared__ int sharedMemory[]; // Tamaño de bloque + 2 para la ventana compartida y tamaño de bloque para almacenar el mejor valor local de cada hilo
    int* sharedWindow = sharedMemory; // La ventana compartida se almacena al inicio de la memoria compartida
    int* sharedLocalBest = sharedMemory + blockDim.x + 2; // Los mejores valores locales se almacenan después de la ventana compartida
    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int identity = isMax ? INT_MIN : INT_MAX;
    int validElementsInBlock = n - blockStart;
    // Ajustar el número de elementos válidos en el bloque para no exceder el tamaño del bloque
    if (validElementsInBlock < 0) {
        validElementsInBlock = 0;
    }
    if (validElementsInBlock > blockDim.x) {
        validElementsInBlock = blockDim.x;
    }
    sharedWindow[localIndex + 1] = globalIndex < n ? data[globalIndex] : identity; // Cargar el elemento actual en la ventana compartida, o el valor identidad si el índice global excede el número de elementos
    if (localIndex == 0) {
        sharedWindow[0] = blockStart > 0 ? data[blockStart - 1] : identity; // Cargar el elemento anterior al bloque en la ventana compartida, o el valor identidad si no hay elemento anterior
        if (validElementsInBlock > 0) {
            const int nextIndex = blockStart + validElementsInBlock;
            sharedWindow[validElementsInBlock + 1] = nextIndex < n ? data[nextIndex] : identity; // Cargar el elemento siguiente al bloque en la ventana compartida, o el valor identidad si no hay elemento siguiente
        } else {
            sharedWindow[1] = identity;
        }
    }

    __syncthreads(); // Sincronizar para asegurarse de que todos los hilos han cargado sus datos en la ventana compartida

    // Cada hilo calcula su mejor valor local comparando su elemento con el anterior y el siguiente en la ventana compartida
    if (globalIndex < n) {
        int bestValue = deviceCompareReduction(sharedWindow[localIndex], sharedWindow[localIndex + 1], isMax);
        bestValue = deviceCompareReduction(bestValue, sharedWindow[localIndex + 2], isMax);
        sharedLocalBest[localIndex] = bestValue;
    } else {
        sharedLocalBest[localIndex] = identity;
    }
    __syncthreads(); // Sincronizar para asegurarse de que todos los hilos han calculado sus mejores valores locales

    // Cada hilo par compara su mejor valor local con el mejor valor local de su hilo vecino impar para obtener un resultado antes de la reducción atómica en memoria global
    if (globalIndex >= n || (globalIndex % 2) != 0) {
        return;
    }
    int pairBest = sharedLocalBest[localIndex]; // El hilo par toma su mejor valor local
    const int nextGlobalIndex = globalIndex + 1;
    if (nextGlobalIndex < n) {
        if (localIndex + 1 < validElementsInBlock) {
            pairBest = deviceCompareReduction(pairBest, sharedLocalBest[localIndex + 1], isMax); // Comparar el mejor valor local del hilo par con el mejor valor local de su hilo vecino impar en la memoria compartida
        } else {
            pairBest = deviceCompareReduction(pairBest, computeWindowReductionFromGlobal(data, n, nextGlobalIndex, isMax), isMax);  // Si el hilo vecino impar no tiene un valor local válido en la memoria compartida, comparar con el resultado de la reducción de ventana para el elemento del hilo vecino impar en memoria global
        }
    }

    if (isMax) {
        atomicMax(result, pairBest); // Reducción atómica para máximo con el mejor valor encontrado entre el hilo par y su hilo vecino impar
    } else {
        atomicMin(result, pairBest); // Reducción atómica para mínimo con el mejor valor encontrado entre el hilo par y su hilo vecino impar
    }
}


/** @brief Realiza una reducción en patrón de árbol utilizando la memoria compartida.
 * Esta función realiza una reducción en patrón de árbol utilizando la memoria compartida para comparar elementos adyacentes y reducir el número de operaciones atómicas necesarias en memoria global.
 * @param input Puntero a los datos de entrada.
 * @param output Puntero al resultado de la reducción.
 * @param n Número de elementos.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 */
__global__ void reductionPattern(const int* input, int* output, int n, bool isMax)
{
    extern __shared__ int sharedReduction[];
    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;
    const int identity = isMax ? INT_MIN : INT_MAX;
    sharedReduction[localIndex] = globalIndex < n ? input[globalIndex] : identity;
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

int getReductionIdentity(bool isMax)
{
    return isMax ? INT_MIN : INT_MAX;
}

int hostCompareReduction(int left, int right, bool isMax)
{
    if (isMax) {
        return left > right ? left : right;
    }

    return left < right ? left : right;
}

bool phase03AtomicVariant(
    Phase3AtomicVariant variant,
    int* deviceInput,
    int totalElements,
    bool isMax,
    const LaunchConfig& launchConfig,
    int& outResult)
{
    int* deviceResult = nullptr;
    std::size_t sharedBytes = 0;
    const int initialValue = getReductionIdentity(isMax);
    cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(int));
    cudaMemcpy(deviceResult, &initialValue, sizeof(int), cudaMemcpyHostToDevice);

    if (variant == Phase3AtomicVariant::Simple) {
        reductionSimple<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
            deviceInput,
            deviceResult,
            totalElements,
            isMax);
    } else if (variant == Phase3AtomicVariant::Basic) {
        sharedBytes = static_cast<std::size_t>(launchConfig.threadsPerBlock + 2) * sizeof(int);
        reductionBasic<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
            deviceInput,
            deviceResult,
            totalElements,
            isMax);
    } else {
        sharedBytes =
            static_cast<std::size_t>(launchConfig.threadsPerBlock + 2 + launchConfig.threadsPerBlock) * sizeof(int);
        reductionIntermediate<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
            deviceInput,
            deviceResult,
            totalElements,
            isMax);
    }
    if (!executeAndWait("Fase 03 atomica")) {
        cudaFree(deviceResult);
        return false;
    }
    cudaMemcpy(&outResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(deviceResult);
    return true;
}

bool phase03ReductionVariant(int* deviceInput, int totalElements, bool isMax, int& outResult)
{
    int* currentInput = deviceInput;
    bool ownsCurrentInput = false;
    int currentCount = totalElements;
    while (currentCount > 10) {
        const LaunchConfig launchConfig = computeLaunchConfig(currentCount);
        const std::size_t partialBytes = static_cast<std::size_t>(launchConfig.blocks) * sizeof(int);
        const std::size_t sharedBytes = static_cast<std::size_t>(launchConfig.threadsPerBlock) * sizeof(int);
        int* devicePartials = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&devicePartials), partialBytes);
        reductionPattern<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
            currentInput,
            devicePartials,
            currentCount,
            isMax);
        if (!executeAndWait("reductionPattern")) {
            cudaFree(devicePartials);
            if (ownsCurrentInput) {
                cudaFree(currentInput);
            }
            return false;
        }
        if (ownsCurrentInput) {
            cudaFree(currentInput);
        }
        currentInput = devicePartials;
        ownsCurrentInput = true;
        currentCount = launchConfig.blocks;
    }
    std::vector<int> hostFinalValues(static_cast<std::size_t>(currentCount));
    cudaMemcpy(
        hostFinalValues.data(),
        currentInput,
        static_cast<std::size_t>(currentCount) * sizeof(int),
        cudaMemcpyDeviceToHost);
    outResult = hostFinalValues[0];
    for (int i = 1; i < currentCount; ++i) {
        outResult = hostCompareReduction(outResult, hostFinalValues[static_cast<std::size_t>(i)], isMax);
    }
    if (ownsCurrentInput) {
        cudaFree(currentInput);
    }
    return true;
}

} // namespace

void phase03(int columnOption, int reductionOption)
{
    const std::vector<float>* sourceColumn = &g_dataset.depDelay;
    const char* columnLabel = "DEP_DELAY";
    if (columnOption == 2) {
        sourceColumn = &g_dataset.arrDelay;
        columnLabel = "ARR_DELAY";
    } else if (columnOption == 3) {
        sourceColumn = &g_dataset.weatherDelay;
        columnLabel = "WEATHER_DELAY";
    }
    const bool isMax = reductionOption == 1;
    const char* reductionLabel = isMax ? "Maximo" : "Minimo";
    const char* reductionFunctionLabel = isMax ? "Max" : "Min";
    std::vector<int> inputValues;
    inputValues.reserve(sourceColumn->size());
    for (std::size_t i = 0; i < sourceColumn->size(); ++i) {
        if (!std::isnan((*sourceColumn)[i])) {
            inputValues.push_back(static_cast<int>((*sourceColumn)[i]));
        }
    }
    const int totalElements = static_cast<int>(inputValues.size());
    if (totalElements <= 0) {
        std::cout << "No hay valores validos para la Fase 03.\n";
        return;
    }
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements);
    std::cout << columnLabel
              << " | " << reductionLabel
              << " | validos " << totalElements
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";
    int* deviceInput = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputValues.size() * sizeof(int));
    cudaMemcpy(deviceInput, inputValues.data(), inputValues.size() * sizeof(int), cudaMemcpyHostToDevice);
    int simpleResult = 0;
    int basicResult = 0;
    int intermediateResult = 0;
    int reductionResult = 0;
    if (!phase03AtomicVariant(Phase3AtomicVariant::Simple, deviceInput, totalElements, isMax, launchConfig, simpleResult) ||
        !phase03AtomicVariant(Phase3AtomicVariant::Basic, deviceInput, totalElements, isMax, launchConfig, basicResult) ||
        !phase03AtomicVariant(Phase3AtomicVariant::Intermediate, deviceInput, totalElements, isMax, launchConfig, intermediateResult) ||
        !phase03ReductionVariant(deviceInput, totalElements, isMax, reductionResult)) {
        cudaFree(deviceInput);
        std::cout << "La Fase 03 no se ha podido completar.\n";
        return;
    }
    cudaFree(deviceInput);
    std::cout << "\n[Simple] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << simpleResult << " minutos\n";
    std::cout << "[Basica] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << basicResult << " minutos\n";
    std::cout << "[Intermedia] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << intermediateResult << " minutos\n";
    std::cout << "[Reduccion] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << reductionResult << " minutos\n";
}
