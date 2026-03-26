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

constexpr int kReductionCpuThreshold = 10; // Umbral para decidir cuándo realizar la reducción final en la CPU después de las reducciones parciales en la GPU
constexpr int kReductionMaxThreads = 256; // Número máximo de hilos por bloque para la reducción en patrón de árbol, ajustado para evitar problemas de rendimiento en GPUs con un número limitado de hilos por bloque

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

/** @brief Reduce la ultima warp en memoria compartida sin sincronizaciones extra.
 * Documentacionde nVidia https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * @param sharedReduction Buffer de reduccion en memoria compartida.
 * @param localIndex Indice local del hilo.
 * @param blockSize Tamano del bloque, asumido potencia de dos.
 * @param isMax Indica si se desea obtener el maximo (true) o el minimo (false).
 */
__device__ void warpReduceShared(volatile int* sharedReduction, unsigned int localIndex, unsigned int blockSize, bool isMax)
{
    // Para bloques de tamaño 64 o más, se necesitan 6 pasos para reducir a un solo valor en la posición 0 del bloque
    if (blockSize >= 64 && localIndex < 32) {
        sharedReduction[localIndex] =
            deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + 32], isMax);
    }
    if (blockSize >= 32 && localIndex < 16) {
        sharedReduction[localIndex] =
            deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + 16], isMax);
    }
    if (blockSize >= 16 && localIndex < 8) {
        sharedReduction[localIndex] =
            deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + 8], isMax);
    }
    if (blockSize >= 8 && localIndex < 4) {
        sharedReduction[localIndex] =
            deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + 4], isMax);
    }
    if (blockSize >= 4 && localIndex < 2) {
        sharedReduction[localIndex] =
            deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + 2], isMax);
    }
    if (blockSize >= 2 && localIndex < 1) {
        sharedReduction[localIndex] =
            deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + 1], isMax);
    }
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


/** @brief Realiza una reducción en patrón paralelo de reduccion utilizando la memoria compartida.
 * 
 * Esta reducción se realiza en varias etapas. En la primera etapa, cada hilo carga un elemento de la memoria global a la memoria compartida y 
 * realiza una comparación con su elemento vecino para obtener un valor parcial. 
 * Luego, se realizan reducciones iterativas en la memoria compartida, donde cada hilo compara su valor parcial con el valor parcial de su vecino a una distancia creciente (2, 4, 8, etc.) 
 * hasta que solo queda un valor parcial por bloque. Finalmente, el hilo 0 de cada bloque escribe el resultado parcial del bloque en la memoria global. 
 * Este proceso se repite hasta que se obtiene el resultado final.
 * 
 * @param input Puntero a los datos de entrada.
 * @param output Puntero al resultado de la reducción.
 * @param n Número de elementos.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 */
__global__ void reductionPattern(const int* input, int* output, int n, bool isMax)
{
    extern __shared__ int sharedReduction[];
    const unsigned int localIndex = threadIdx.x;
    const unsigned int baseIndex = blockIdx.x * (blockDim.x * 2) + localIndex;
    const int identity = isMax ? INT_MIN : INT_MAX;

    int localBest = identity;

    // Cada hilo carga un elemento de la memoria global a la memoria compartida y realiza una comparación con su elemento vecino para obtener un valor parcial
    if (baseIndex < static_cast<unsigned int>(n)) {
        localBest = input[baseIndex];
    }

    // Comparar con el elemento vecino a la derecha (si existe) para obtener un valor parcial
    const unsigned int pairedIndex = baseIndex + blockDim.x;
    if (pairedIndex < static_cast<unsigned int>(n)) {
        localBest = deviceCompareReduction(localBest, input[pairedIndex], isMax);
    }

    sharedReduction[localIndex] = localBest;
    __syncthreads(); // Sincronizar para asegurarse de que todos los hilos han cargado sus valores parciales en la memoria compartida

    // Realizar reducciones iterativas en la memoria compartida, donde cada hilo compara su valor parcial con el valor parcial de su vecino a una distancia creciente (2, 4, 8, etc.) 
    // hasta que solo queda un valor parcial por bloque
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (localIndex < stride) {
            sharedReduction[localIndex] =
                deviceCompareReduction(sharedReduction[localIndex], sharedReduction[localIndex + stride], isMax); // Comparar el valor parcial del hilo con el valor parcial de su vecino a una distancia creciente en la memoria compartida
        }
        __syncthreads(); //Esperamos y sincronizamos :)
    }

    // Realizar la reducción final de la última warp sin sincronizaciones extra
    if (localIndex < 32) {
        warpReduceShared(sharedReduction, localIndex, blockDim.x, isMax);
    }

    if (localIndex == 0) {
        output[blockIdx.x] = sharedReduction[0];
    }
}

/** @brief Obtiene el valor identidad para la reducción.
 * 
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 * @return El valor identidad para la reducción.
 */
int getReductionIdentity(bool isMax)
{
    return isMax ? INT_MIN : INT_MAX;
}

/** @brief Compara dos valores para la reducción.
 * 
 * @param left Primer valor.
 * @param right Segundo valor.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 * @return El valor comparado.
 */
int hostCompareReduction(int left, int right, bool isMax)
{
    if (isMax) {
        return left > right ? left : right;
    }

    return left < right ? left : right;
}

/** @brief Calcula la mayor potencia de dos menor o igual a un valor dado.
 * 
 * @param value El valor para el cual se desea calcular la potencia de dos.
 * @return La mayor potencia de dos menor o igual al valor dado.
 */
int floorPowerOfTwo(int value)
{
    int power = 1;

    while ((power << 1) > 0 && (power << 1) <= value) {
        power <<= 1;
    }

    return power;
}

/** @brief Calcula la configuración de lanzamiento para la reducción en patrón paralelo.
 * 
 * Esta función calcula el número de bloques y el número de hilos por bloque para lanzar el kernel de reducción en patrón paralelo, 
 * teniendo en cuenta el número total de elementos a reducir y las limitaciones de la GPU. 
 * El número de hilos por bloque se ajusta a la mayor potencia de dos menor o igual al número máximo de hilos por bloque permitido por la GPU, 
 * con un límite adicional para evitar problemas de rendimiento en GPUs con un número limitado de hilos por bloque. 
 * El número de bloques se calcula en función del número total de elementos y el número de elementos que cada bloque puede procesar (hilos por bloque * 2).
 * 
 * @param totalElements El número total de elementos a reducir.
 * @return La configuración de lanzamiento calculada para la reducción en patrón paralelo.
 */
LaunchConfig computeReductionLaunchConfig(int totalElements)
{
    LaunchConfig launchConfig;

    if (totalElements <= 0) {
        return launchConfig;
    }

    const int maxThreads = g_deviceProp.maxThreadsPerBlock > 0 ? g_deviceProp.maxThreadsPerBlock : 1; // Ajustar el número de hilos por bloque a la mayor potencia de dos menor o igual al número máximo de hilos por bloque permitido por la GPU, con un límite adicional para evitar problemas de rendimiento en GPUs con un número limitado de hilos por bloque
    const int cappedThreads = maxThreads < kReductionMaxThreads ? maxThreads : kReductionMaxThreads; // Limitar el número de hilos por bloque a kReductionMaxThreads para evitar problemas de rendimiento en GPUs con un número limitado de hilos por bloque

    launchConfig.threadsPerBlock = floorPowerOfTwo(cappedThreads); // Asegurarse de que el número de hilos por bloque es al menos 1
    if (launchConfig.threadsPerBlock <= 0) {
        launchConfig.threadsPerBlock = 1;
    }

    const int elementsPerBlock = launchConfig.threadsPerBlock * 2; // Cada bloque puede procesar hilos por bloque * 2 elementos debido a la estrategia de reducción en patrón paralelo
    launchConfig.blocks = (totalElements + elementsPerBlock - 1) / elementsPerBlock; // Calcular el número de bloques necesario para procesar todos los elementos, redondeando hacia arriba
    return launchConfig;
}

/** @brief Ejecuta una variante de reducción atómica y obtiene el resultado.
 * 
 * Esta función ejecuta una de las variantes de reducción atómica (Simple, Básica o Intermedia) en la GPU, 
 * utilizando la configuración de lanzamiento proporcionada. Después de ejecutar el kernel, se espera a que la GPU termine la ejecución y se copia el resultado de la reducción desde la memoria global a la memoria host. 
 * Finalmente, se libera la memoria utilizada para el resultado en la GPU.
 * 
 * @param variant La variante de reducción atómica a ejecutar (Simple, Básica o Intermedia).
 * @param deviceInput Puntero a los datos de entrada en la memoria global de la GPU.
 * @param totalElements El número total de elementos a reducir.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 * @param launchConfig La configuración de lanzamiento para el kernel de reducción.
 * @param outResult Referencia para almacenar el resultado de la reducción después de copiarlo desde la GPU.
 * @return true si la ejecución y obtención del resultado fue exitosa, false en caso contrario.
 */
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
    const int initialValue = getReductionIdentity(isMax); // Obtener el valor identidad para la reducción (INT_MIN para máximo, INT_MAX para mínimo)
    cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(int)); // Reservar memoria en la GPU para el resultado de la reducción
    cudaMemcpy(deviceResult, &initialValue, sizeof(int), cudaMemcpyHostToDevice); // Copiar el valor identidad a la memoria de la GPU para inicializar el resultado de la reducción

    // Ejecutar la variante de reducción atómica seleccionada en la GPU utilizando la configuración de lanzamiento proporcionada
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
    if (!executeAndWait("Fase 03 atomica")) { // Esperar a que la GPU termine la ejecución del kernel y verificar si hubo errores
        cudaFree(deviceResult);
        return false;
    }
    cudaMemcpy(&outResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost); // Copiar el resultado de la reducción desde la memoria global de la GPU a la memoria host
    cudaFree(deviceResult);                                                    // Liberar la memoria utilizada para el resultado en la GPU. be free my dear memory ;)
    return true;
}

/** @brief Realiza una reducción utilizando un patrón paralelo de reducción en la GPU.
 * 
 * Esta función realiza una reducción utilizando un patrón paralelo de reducción en la GPU, donde se lanzan múltiples kernels de reducción en patrón paralelo de manera iterativa hasta que el número de elementos a reducir es menor o igual a un umbral definido (kReductionCpuThreshold). 
 * En cada iteración, se calcula la configuración de lanzamiento para el kernel de reducción en patrón paralelo, se reserva memoria para los resultados parciales, se ejecuta el kernel y se espera a que termine la ejecución. 
 * Después de cada kernel, se actualiza el puntero a los datos de entrada para la siguiente iteración con los resultados parciales obtenidos. 
 * Finalmente, cuando el número de elementos a reducir es menor o igual al umbral, se copian los resultados parciales restantes a la memoria host y se realiza la reducción final en la CPU para obtener el resultado final.
 * 
 * @param deviceInput Puntero a los datos de entrada en la memoria global de la GPU.
 * @param totalElements El número total de elementos a reducir.
 * @param isMax Indica si se desea obtener el máximo (true) o el mínimo (false).
 * @param outResult Referencia para almacenar el resultado final de la reducción después de copiarlo desde la GPU y realizar la reducción final en la CPU.
 * @return true si la reducción se realizó exitosamente, false en caso contrario.
 */
bool phase03ReductionVariant(int* deviceInput, int totalElements, bool isMax, int& outResult)
{
    int* currentInput = deviceInput; // Puntero a los datos de entrada para la iteración actual, inicialmente apunta a los datos de entrada originales en la GPU
    bool ownsCurrentInput = false; // Indica si la función es responsable de liberar la memoria apuntada por currentInput, inicialmente es false porque no se ha reservado memoria adicional para los resultados parciales
    int currentCount = totalElements; 

    // Realizar reducciones iterativas en la GPU utilizando el patrón paralelo de reducción hasta que el número de elementos a reducir sea menor o igual al umbral definido (kReductionCpuThreshold)
    while (currentCount > kReductionCpuThreshold) {
        const LaunchConfig launchConfig = computeReductionLaunchConfig(currentCount); // Calcular la configuración de lanzamiento para el kernel de reducción en patrón paralelo en función del número actual de elementos a reducir
        const std::size_t partialBytes = static_cast<std::size_t>(launchConfig.blocks) * sizeof(int);
        const std::size_t sharedBytes = static_cast<std::size_t>(launchConfig.threadsPerBlock) * sizeof(int);
        int* devicePartials = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&devicePartials), partialBytes); // Reservar memoria en la GPU para los resultados parciales de la reducción de esta iteración
        reductionPattern<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>( 
            currentInput,
            devicePartials,
            currentCount,
            isMax);
        if (!executeAndWait("reductionPattern")) { // Esperar a que la GPU termine la ejecución del kernel de reducción en patrón paralelo y verificar si hubo errores
            cudaFree(devicePartials);
            if (ownsCurrentInput) {
                cudaFree(currentInput);
            }
            return false;
        }
        if (ownsCurrentInput) { // Si la función es responsable de liberar la memoria apuntada por currentInput, liberar esa memoria antes de actualizar el puntero a los datos de entrada para la siguiente iteración
            cudaFree(currentInput);
        }
        currentInput = devicePartials;
        ownsCurrentInput = true;
        currentCount = launchConfig.blocks;
    }
    std::vector<int> hostFinalValues(static_cast<std::size_t>(currentCount)); // Vector para almacenar los resultados parciales restantes después de las reducciones iterativas en la GPU, que serán copiados a la memoria host para realizar la reducción final en la CPU
    cudaMemcpy(
        hostFinalValues.data(),
        currentInput,
        static_cast<std::size_t>(currentCount) * sizeof(int),
        cudaMemcpyDeviceToHost);
    outResult = hostFinalValues[0];

    // Realizar la reducción final en la CPU para obtener el resultado final, comparando los resultados parciales restantes copiados a la memoria host
    for (int i = 1; i < currentCount; ++i) {
        outResult = hostCompareReduction(outResult, hostFinalValues[static_cast<std::size_t>(i)], isMax);
    }
    if (ownsCurrentInput) {
        cudaFree(currentInput);
    }
    return true;
}

} // namespace

/** @brief Función principal para la fase 03, que realiza reducciones atómicas y en patrón paralelo para obtener el máximo o mínimo de una columna específica del dataset.
 * 
 * Esta función selecciona la columna del dataset a procesar según el parámetro columnOption, determina si se desea obtener el máximo o mínimo según el parámetro reductionOption, y prepara los datos de entrada para la reducción. 
 * Luego, ejecuta las variantes de reducción atómica (Simple, Básica e Intermedia) y la reducción en patrón paralelo, obteniendo los resultados de cada una. Finalmente, imprime los resultados obtenidos para cada variante de reducción.
 * 
 * @param columnOption Opción para seleccionar la columna del dataset (1 para DEP_DELAY, 2 para ARR_DELAY, 3 para WEATHER_DELAY).
 * @param reductionOption Opción para seleccionar el tipo de reducción (1 para máximo, 2 para mínimo).
 */
void phase03(int columnOption, int reductionOption)
{
    const std::vector<float>* sourceColumn = &g_dataset.depDelay;
    const char* columnLabel = "DEP_DELAY"; // Columna por defecto es DEP_DELAY
    // Seleccionar la columna del dataset a procesar según el parámetro columnOption
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
    inputValues.reserve(sourceColumn->size()); // Reservar memoria para los valores de entrada, asumiendo que la mayoría de los valores serán válidos (no NaN)
    for (std::size_t i = 0; i < sourceColumn->size(); ++i) {
        if (!std::isnan((*sourceColumn)[i])) {
            inputValues.push_back(static_cast<int>((*sourceColumn)[i])); // Convertir los valores de la columna a enteros para la reducción, asumiendo que los valores son tiempos de retraso en minutos y se pueden representar como enteros
        }
    }
    const int totalElements = static_cast<int>(inputValues.size());
    if (totalElements <= 0) {
        std::cout << "No hay valores validos para la Fase 03.\n";
        return;
    }
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements); // Calcular la configuración de lanzamiento para las variantes de reducción atómica en función del número total de elementos a reducir
    std::cout << columnLabel
              << " | " << reductionLabel
              << " | validos " << totalElements
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";
    int* deviceInput = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputValues.size() * sizeof(int)); // Reservar memoria en la GPU para los datos de entrada de la reducción
    cudaMemcpy(deviceInput, inputValues.data(), inputValues.size() * sizeof(int), cudaMemcpyHostToDevice); // Copiar los datos de entrada desde la memoria host a la memoria global de la GPU para la reducción
    int simpleResult = 0;
    int basicResult = 0;
    int intermediateResult = 0;
    int reductionResult = 0;

    // Ejecutar las variantes de reducción atómica (Simple, Básica e Intermedia) y la reducción en patrón paralelo, obteniendo los resultados de cada una. 
    // Si alguna de las ejecuciones falla, liberar la memoria utilizada para los datos de entrada y salir de la función.
    if (!phase03AtomicVariant(Phase3AtomicVariant::Simple, deviceInput, totalElements, isMax, launchConfig, simpleResult) ||
        !phase03AtomicVariant(Phase3AtomicVariant::Basic, deviceInput, totalElements, isMax, launchConfig, basicResult) ||
        !phase03AtomicVariant(Phase3AtomicVariant::Intermediate, deviceInput, totalElements, isMax, launchConfig, intermediateResult) ||
        !phase03ReductionVariant(deviceInput, totalElements, isMax, reductionResult)) {
        cudaFree(deviceInput);
        std::cout << "La Fase 03 no se ha podido completar.\n";
        return;
    }
    // Imprimir los resultados obtenidos para cada variante de reducción y liberar la memoria utilizada para los datos de entrada en la GPU. 
    cudaFree(deviceInput); // be free my dear input memory ;)
    std::cout << "\n[Simple] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << simpleResult << " minutos\n";
    std::cout << "[Basica] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << basicResult << " minutos\n";
    std::cout << "[Intermedia] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << intermediateResult << " minutos\n";
    std::cout << "[Reduccion] " << reductionFunctionLabel << "() " << columnLabel
              << " = " << reductionResult << " minutos\n";
}
