#include "comun.cuh"

#include <iostream>

/*
    comun.cu

    Implementa el estado global compartido y las utilidades CUDA comunes.
*/

DatasetColumns g_dataset;                       // Estructura que contiene las columnas del dataset cargado
LoadSummary g_summary;                          // Estructura que contiene el resumen de la carga del dataset
std::string g_datasetPath;                      // Ruta del dataset cargado, se mantiene para mostrarla en el resumen de la GPU
bool g_datasetLoaded = false;                   // Indica si el dataset se ha cargado correctamente, se utiliza para mostrar información relevante en el resumen de la GPU

bool g_deviceReady = false;                     // Indica si la GPU está lista para ejecutar kernels, se establece después de consultar la información de la GPU    
cudaDeviceProp g_deviceProp{};                  // Estructura que contiene las propiedades de la GPU, se llena al consultar la información de la GPU
std::string g_deviceErrorMessage;               // Mensaje de error relacionado con la GPU, se establece si ocurre un error al consultar la información de la GPU

int g_rowCount = 0;                             // Número total de filas del dataset, se mantiene para mostrarlo en el resumen de la GPU y para calcular la configuración de lanzamiento base

float* d_depDelay = nullptr;                    // Puntero a la columna de retrasos de salida en la GPU
float* d_arrDelay = nullptr;                    // Puntero a la columna de retrasos de llegada en la GPU
char* d_tailNums = nullptr;                     // Puntero a la columna de tailNums en la GPU, se mantiene como char* para facilitar el manejo de strings de longitud variable
int* d_originSeqId = nullptr;                   // Puntero a la columna de originSeqId en la GPU
int* d_destSeqId = nullptr;                     // Puntero a la columna de destSeqId en la GPU
int* d_originAirportCodes = nullptr;            // Puntero a la columna de códigos de aeropuerto de origen en la GPU, se mantiene como int* para facilitar el manejo de códigos de aeropuerto mapeados a enteros
int* d_destAirportCodes = nullptr;              // Puntero a la columna de códigos de aeropuerto de destino en la GPU, se mantiene como int* para facilitar el manejo de códigos de aeropuerto mapeados a enteros

int* d_phase2Count = nullptr;                   // Puntero a la variable que almacena el conteo de filas que cumplen la condición en la Fase 02, se mantiene como int* para facilitar su actualización desde el kernel
int* d_phase2OutDelayValues = nullptr;          // Puntero a la columna de valores de retraso de salida que cumplen la condición en la Fase 02, se mantiene como float* para facilitar su manejo como valores numéricos
char* d_phase2OutTailNums = nullptr;            // Puntero a la columna de tailNums que cumplen la condición en la Fase 02, se mantiene como char* para facilitar el manejo de strings de longitud variable

int* d_originDenseInput = nullptr;              // Puntero a la columna de originSeqId mapeada a IDs densos en la GPU, se mantiene como int* para facilitar el manejo de IDs densos
int g_originTotalElements = 0;                  // Número total de elementos en la columna de originSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
int g_originTotalBins = 0;                      // Número total de bins únicos en la columna de originSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
std::vector<int> g_originDenseToSeqId;          // Vector que mapea los IDs densos de originSeqId a los IDs originales

int* d_destinationDenseInput = nullptr;         // Puntero a la columna de destSeqId mapeada a IDs densos en la GPU, se mantiene como int* para facilitar el manejo de IDs densos       
int g_destinationTotalElements = 0;             // Número total de elementos en la columna de destSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
int g_destinationTotalBins = 0;                 // Número total de bins únicos en la columna de destSeqId, se mantiene para calcular la configuración de lanzamiento y para mostrarlo en el resumen de la GPU
std::vector<int> g_destinationDenseToSeqId;     // Vector que mapea los IDs densos de destSeqId a los IDs originales

/**
 * Ejecuta un kernel CUDA y espera a que termine.
 * @param context Descripción del contexto donde se ejecuta el kernel.
 * @return true si la ejecución es exitosa, false en caso contrario.
 */
bool executeAndWait(const char* context)
{
    const cudaError_t launchStatus = cudaGetLastError(); // Verificar errores de lanzamiento del kernel

    // Si el lanzamiento del kernel fue exitoso, esperar a que termine y verificar errores de ejecución
    if (launchStatus != cudaSuccess) {
        std::cout << "Error CUDA en " << context << ": "
                  << cudaGetErrorString(launchStatus) << "\n";
        return false;
    }

    const cudaError_t syncStatus = cudaDeviceSynchronize(); // Esperar a que el kernel termine y verificar errores de ejecución

    if (syncStatus != cudaSuccess) {
        std::cout << "Error CUDA en " << context << ": "
                  << cudaGetErrorString(syncStatus) << "\n";
        return false;
    }

    return true;
}

/**
 * Consulta la información de la GPU.
 * @return true si la consulta es exitosa, false en caso contrario.
 */
bool queryGpuInfo()
{
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount); // Verificar errores al obtener el conteo de dispositivos CUDA

    // Si no se pudo obtener el conteo de dispositivos, marcar la GPU como no lista y almacenar el mensaje de error
    if (countStatus != cudaSuccess) {
        g_deviceReady = false;
        g_deviceErrorMessage = cudaGetErrorString(countStatus);
        return false;
    }

    if (deviceCount <= 0) {
        g_deviceReady = false;
        g_deviceErrorMessage = "No se ha detectado ninguna GPU CUDA accesible.";
        return false;
    }

    const cudaError_t propertyStatus = cudaGetDeviceProperties(&g_deviceProp, 0); // Verificar errores al obtener las propiedades del dispositivo

    // Si no se pudieron obtener las propiedades del dispositivo, marcar la GPU como no lista y almacenar el mensaje de error
    if (propertyStatus != cudaSuccess) {
        g_deviceReady = false;
        g_deviceErrorMessage = cudaGetErrorString(propertyStatus);
        return false;
    }

    g_deviceReady = true;
    g_deviceErrorMessage.clear();
    return true;
}

/**
 * Calcula la configuración de lanzamiento para un kernel CUDA.
 * Para el cálculo se toma como base el número total de elementos a procesar y las capacidades del dispositivo.
 * De acuerdo a esto se determina un número de hilos por bloque (hasta un máximo de 256) y el número de bloques necesarios para cubrir todos los elementos.
 * @param totalElements Número total de elementos a procesar.
 * @return La configuración de lanzamiento.
 */
LaunchConfig computeLaunchConfig(int totalElements) 
{
    LaunchConfig launchConfig; // Inicializar con valores predeterminados
    const int maxThreads = g_deviceProp.maxThreadsPerBlock > 0 ? g_deviceProp.maxThreadsPerBlock : 1; // Asegurar que el número máximo de hilos por bloque sea al menos 1

    // Determinar el número de hilos por bloque, con un máximo de 256
    launchConfig.threadsPerBlock = maxThreads < 256 ? maxThreads : 256;

    if (totalElements > 0) {
        // Calcular el número de bloques necesarios para cubrir todos los elementos, redondeando hacia arriba
        launchConfig.blocks = (totalElements + launchConfig.threadsPerBlock - 1) / launchConfig.threadsPerBlock;
    }

    return launchConfig;
}

/**
 * Imprime un resumen de la información de la GPU. Incluye:
 * - El nombre de la GPU y su capacidad de cómputo.
 * - La memoria global total, la memoria compartida por bloque y el número máximo de hilos por bloque.
 * - Si el dataset está cargado, sugiere una configuración de lanzamiento base y confirma que el dataset está cargado en la GPU para las fases 01, 02 y 04.
 * Si la GPU no está lista, muestra el mensaje de error correspondiente. 
 */
void printGpuSummary()
{
    std::cout << "\n=== CUDA ===\n";

    if (!g_deviceReady) {
        std::cout << "No disponible: " << g_deviceErrorMessage << "\n";
        return;
    }

    // Calcular la memoria global en MB y la memoria compartida por bloque en KB para una presentación más legible
    const unsigned long long globalMemoryInMb =
        static_cast<unsigned long long>(g_deviceProp.totalGlobalMem) / (1024ULL * 1024ULL); // dividir por 1024 dos veces para convertir de bytes a MB
    const unsigned long long sharedMemoryInKb =
        static_cast<unsigned long long>(g_deviceProp.sharedMemPerBlock) / 1024ULL; // dividir por 1024 para convertir de bytes a KB

    std::cout << "GPU: " << g_deviceProp.name
              << " | CC " << g_deviceProp.major << "." << g_deviceProp.minor << "\n"; // Mostrar el nombre de la GPU y su capacidad de cómputo (CC)
    std::cout << "Global: " << globalMemoryInMb << " MB"
              << " | Shared por bloque: " << sharedMemoryInKb << " KB"
              << " | Max hilos/bloque: " << g_deviceProp.maxThreadsPerBlock << "\n"; 

    if (g_datasetLoaded) {
        const LaunchConfig launchConfig = computeLaunchConfig(static_cast<int>(g_dataset.depDelay.size())); // Calcular la configuración de lanzamiento base utilizando el número total de elementos en depDelay

        std::cout << "Sugerencia base: " << launchConfig.blocks
                  << " bloques x " << launchConfig.threadsPerBlock << " hilos\n";

        if (g_rowCount > 0) {
            std::cout << "Dataset cargado en GPU para Fases 01, 02 y 04.\n";
        }
    }
}
