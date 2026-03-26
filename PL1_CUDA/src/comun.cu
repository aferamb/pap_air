#include "comun.cuh"

#include <iostream>

/*
    comun.cu

    Implementa el estado global compartido y las utilidades CUDA comunes.
*/

DatasetColumns g_dataset;
LoadSummary g_summary;
std::string g_datasetPath;
bool g_datasetLoaded = false;

bool g_deviceReady = false;
cudaDeviceProp g_deviceProp{};
std::string g_deviceErrorMessage;

int g_rowCount = 0;

float* d_depDelay = nullptr;
float* d_arrDelay = nullptr;
char* d_tailNums = nullptr;

int* d_phase2Count = nullptr;
int* d_phase2OutDelayValues = nullptr;
char* d_phase2OutTailNums = nullptr;

int* d_originDenseInput = nullptr;
int g_originTotalElements = 0;
int g_originTotalBins = 0;
std::vector<int> g_originDenseToSeqId;

int* d_destinationDenseInput = nullptr;
int g_destinationTotalElements = 0;
int g_destinationTotalBins = 0;
std::vector<int> g_destinationDenseToSeqId;

bool executeAndWait(const char* context)
{
    const cudaError_t launchStatus = cudaGetLastError();

    if (launchStatus != cudaSuccess) {
        std::cout << "Error CUDA en " << context << ": "
                  << cudaGetErrorString(launchStatus) << "\n";
        return false;
    }

    const cudaError_t syncStatus = cudaDeviceSynchronize();

    if (syncStatus != cudaSuccess) {
        std::cout << "Error CUDA en " << context << ": "
                  << cudaGetErrorString(syncStatus) << "\n";
        return false;
    }

    return true;
}

bool queryGpuInfo()
{
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);

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

    const cudaError_t propertyStatus = cudaGetDeviceProperties(&g_deviceProp, 0);

    if (propertyStatus != cudaSuccess) {
        g_deviceReady = false;
        g_deviceErrorMessage = cudaGetErrorString(propertyStatus);
        return false;
    }

    g_deviceReady = true;
    g_deviceErrorMessage.clear();
    return true;
}

LaunchConfig computeLaunchConfig(int totalElements)
{
    LaunchConfig launchConfig;
    const int maxThreads = g_deviceProp.maxThreadsPerBlock > 0 ? g_deviceProp.maxThreadsPerBlock : 1;

    launchConfig.threadsPerBlock = maxThreads < 256 ? maxThreads : 256;

    if (totalElements > 0) {
        launchConfig.blocks = (totalElements + launchConfig.threadsPerBlock - 1) / launchConfig.threadsPerBlock;
    }

    return launchConfig;
}

void printGpuSummary()
{
    std::cout << "\n=== CUDA ===\n";

    if (!g_deviceReady) {
        std::cout << "No disponible: " << g_deviceErrorMessage << "\n";
        return;
    }

    const unsigned long long globalMemoryInMb =
        static_cast<unsigned long long>(g_deviceProp.totalGlobalMem) / (1024ULL * 1024ULL);
    const unsigned long long sharedMemoryInKb =
        static_cast<unsigned long long>(g_deviceProp.sharedMemPerBlock) / 1024ULL;

    std::cout << "GPU: " << g_deviceProp.name
              << " | CC " << g_deviceProp.major << "." << g_deviceProp.minor << "\n";
    std::cout << "Global: " << globalMemoryInMb << " MB"
              << " | Shared por bloque: " << sharedMemoryInKb << " KB"
              << " | Max hilos/bloque: " << g_deviceProp.maxThreadsPerBlock << "\n";

    if (g_datasetLoaded) {
        const LaunchConfig launchConfig = computeLaunchConfig(static_cast<int>(g_dataset.depDelay.size()));

        std::cout << "Sugerencia base: " << launchConfig.blocks
                  << " bloques x " << launchConfig.threadsPerBlock << " hilos\n";

        if (g_rowCount > 0) {
            std::cout << "Dataset cargado en GPU para Fases 01, 02 y 04.\n";
        }
    }
}
