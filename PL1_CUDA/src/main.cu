#include <cuda_runtime.h>

#include <cctype>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "csv_reader.h"
#include "kernels.cuh"

/*
    main.cu

    Esta version reduce bastante el volumen de codigo de host:

    - la CLI se ha compactado;
    - la Fase 0 devuelve solo dataset + resumen;
    - Fases 01, 02 y 04 reutilizan buffers persistentes en GPU;
    - se han eliminado varios helpers pequenos que solo fragmentaban el flujo.

    La parte obligatoria de la practica se mantiene:

    - Fase 01 filtra DEP_DELAY en GPU;
    - Fase 02 filtra ARR_DELAY + TAIL_NUM en GPU y devuelve resultados al host;
    - Fase 03 ejecuta las 4 variantes pedidas;
    - Fase 04 genera el histograma por SEQ_ID en GPU y lo dibuja en CPU.
*/

enum class MainMenuOption {
    Phase1,
    Phase2,
    Phase3,
    Phase4,
    ReloadCsv,
    ShowStatus,
    Exit
};

enum class Phase3ColumnOption {
    DepartureDelay = 1,
    ArrivalDelay = 2,
    WeatherDelay = 3
};

enum class ReductionTypeOption {
    Maximum = 1,
    Minimum = 2
};

enum class HistogramAirportTypeOption {
    Origin = 1,
    Destination = 2
};

enum class DelayFilterMode {
    Delay = 1,
    Advance = 2,
    Both = 3
};

/*
    LaunchConfig

    Estructura minima para describir la configuracion de lanzamiento que se
    usa o se muestra en consola.
*/
struct LaunchConfig {
    int blocks = 0;
    int threadsPerBlock = 1;
};

/*
    Phase4Cache

    Cache persistente especifica de la Fase 04:

    - d_denseInput: indices densos listos para el histograma en GPU;
    - totalElements: numero de filas validas para ese histograma;
    - totalBins: numero de aeropuertos unicos por SEQ_ID;
    - denseToSeqId: traduccion bin -> SEQ_ID para imprimir en CPU.
*/
struct Phase4Cache {
    int* d_denseInput = nullptr;
    int totalElements = 0;
    int totalBins = 0;
    std::vector<int> denseToSeqId;
};

/*
    GpuCache

    Cache persistente de device. Se construye una sola vez al cargar el CSV y
    luego se reutiliza en las fases que mas repiten copias host -> device.
*/
struct GpuCache {
    int rowCount = 0;
    float* d_depDelay = nullptr;
    float* d_arrDelay = nullptr;
    char* d_tailNums = nullptr;
    int* d_phase2Count = nullptr;
    int* d_phase2OutDelayValues = nullptr;
    char* d_phase2OutTailNums = nullptr;
    Phase4Cache origin;
    Phase4Cache destination;
};

/*
    AppState

    Estado vivo de la aplicacion. La idea es guardar solo lo que el menu y las
    fases necesitan de verdad:

    - ruta activa;
    - dataset cargado en host;
    - resumen de la Fase 0;
    - disponibilidad de CUDA;
    - propiedades de la GPU;
    - cache persistente en GPU.
*/
struct AppState {
    std::string datasetPath;
    bool datasetLoaded = false;
    DatasetColumns dataset;
    LoadSummary summary;

    bool deviceReady = false;
    cudaDeviceProp deviceProp{};
    std::string deviceErrorMessage;
    GpuCache gpu;
};

namespace {

enum class Phase3AtomicVariant {
    Simple,
    Basic,
    Intermediate
};

std::string trimWhitespace(const std::string& text)
{
    std::size_t start = 0;
    std::size_t end = text.size();

    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(start, end - start);
}

bool fileExists(const std::string& path)
{
    std::ifstream file(path.c_str());
    return file.good();
}

bool isCancelToken(const std::string& input)
{
    return input == "x" || input == "X";
}

bool cudaOk(cudaError_t status, const char* context)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cout << "Error CUDA en " << context << ": "
              << cudaGetErrorString(status) << "\n";
    return false;
}

bool queryGpuInfo(cudaDeviceProp& deviceProp, std::string& errorMessage)
{
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);

    if (countStatus != cudaSuccess) {
        errorMessage = cudaGetErrorString(countStatus);
        return false;
    }

    if (deviceCount <= 0) {
        errorMessage = "No se ha detectado ninguna GPU CUDA accesible.";
        return false;
    }

    const cudaError_t propertyStatus = cudaGetDeviceProperties(&deviceProp, 0);

    if (propertyStatus != cudaSuccess) {
        errorMessage = cudaGetErrorString(propertyStatus);
        return false;
    }

    return true;
}

LaunchConfig computeLaunchConfig(int totalElements, const cudaDeviceProp& deviceProp)
{
    LaunchConfig launchConfig;

    const int maxThreads = deviceProp.maxThreadsPerBlock > 0 ? deviceProp.maxThreadsPerBlock : 1;
    launchConfig.threadsPerBlock = maxThreads < 256 ? maxThreads : 256;

    if (totalElements > 0) {
        launchConfig.blocks = (totalElements + launchConfig.threadsPerBlock - 1) / launchConfig.threadsPerBlock;
    }

    return launchConfig;
}

const char* getDelayModeLabel(DelayFilterMode mode)
{
    switch (mode) {
    case DelayFilterMode::Delay:
        return "retraso";
    case DelayFilterMode::Advance:
        return "adelanto";
    case DelayFilterMode::Both:
    default:
        return "ambos";
    }
}

const char* getDetectedDelayLabel(int value, DelayFilterMode mode, int threshold)
{
    if (mode == DelayFilterMode::Delay) {
        return "Retraso";
    }

    if (mode == DelayFilterMode::Advance) {
        return "Adelanto";
    }

    return value >= threshold ? "Retraso" : "Adelanto";
}

const char* getAirportLabel(HistogramAirportTypeOption airportType)
{
    return airportType == HistogramAirportTypeOption::Origin ? "origen" : "destino";
}

const char* getPhase3ColumnLabel(Phase3ColumnOption columnOption)
{
    switch (columnOption) {
    case Phase3ColumnOption::ArrivalDelay:
        return "ARR_DELAY";
    case Phase3ColumnOption::WeatherDelay:
        return "WEATHER_DELAY";
    case Phase3ColumnOption::DepartureDelay:
    default:
        return "DEP_DELAY";
    }
}

const char* getReductionLabel(ReductionTypeOption reductionOption)
{
    return reductionOption == ReductionTypeOption::Maximum ? "Maximo" : "Minimo";
}

const char* getReductionFunctionLabel(ReductionTypeOption reductionOption)
{
    return reductionOption == ReductionTypeOption::Maximum ? "Max" : "Min";
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

void buildTailBuffer(const std::vector<std::string>& source, std::vector<char>& outBuffer)
{
    outBuffer.assign(source.size() * kPhase2TailNumStride, '\0');

    for (std::size_t row = 0; row < source.size(); ++row) {
        const std::string& tailNum = source[row];
        char* cell = &outBuffer[row * kPhase2TailNumStride];

        const std::size_t maxCharacters = static_cast<std::size_t>(kPhase2TailNumStride - 1);
        const std::size_t charactersToCopy = tailNum.size() < maxCharacters ? tailNum.size() : maxCharacters;

        for (std::size_t i = 0; i < charactersToCopy; ++i) {
            cell[i] = tailNum[i];
        }

        cell[charactersToCopy] = '\0';
    }
}

void buildPhase4DenseInput(
    const std::vector<int>& seqIds,
    const std::unordered_map<int, std::string>& idToCode,
    std::vector<int>& denseToSeqId,
    std::vector<int>& denseInput)
{
    denseToSeqId.clear();
    denseInput.clear();
    denseInput.reserve(seqIds.size());

    std::unordered_map<int, int> denseIndexBySeqId;

    for (std::size_t i = 0; i < seqIds.size(); ++i) {
        const int seqId = seqIds[i];

        if (seqId < 0) {
            continue;
        }

        if (idToCode.find(seqId) == idToCode.end()) {
            continue;
        }

        std::unordered_map<int, int>::const_iterator found = denseIndexBySeqId.find(seqId);

        if (found == denseIndexBySeqId.end()) {
            const int nextDenseIndex = static_cast<int>(denseToSeqId.size());
            denseIndexBySeqId[seqId] = nextDenseIndex;
            denseToSeqId.push_back(seqId);
            denseInput.push_back(nextDenseIndex);
        } else {
            denseInput.push_back(found->second);
        }
    }
}

void releaseGpuCache(GpuCache& cache)
{
    if (cache.d_depDelay != nullptr) {
        cudaFree(cache.d_depDelay);
    }

    if (cache.d_arrDelay != nullptr) {
        cudaFree(cache.d_arrDelay);
    }

    if (cache.d_tailNums != nullptr) {
        cudaFree(cache.d_tailNums);
    }

    if (cache.d_phase2Count != nullptr) {
        cudaFree(cache.d_phase2Count);
    }

    if (cache.d_phase2OutDelayValues != nullptr) {
        cudaFree(cache.d_phase2OutDelayValues);
    }

    if (cache.d_phase2OutTailNums != nullptr) {
        cudaFree(cache.d_phase2OutTailNums);
    }

    if (cache.origin.d_denseInput != nullptr) {
        cudaFree(cache.origin.d_denseInput);
    }

    if (cache.destination.d_denseInput != nullptr) {
        cudaFree(cache.destination.d_denseInput);
    }

    cache = GpuCache{};
}

bool buildGpuCache(const DatasetColumns& dataset, GpuCache& cache, std::string& errorMessage)
{
    GpuCache newCache;
    newCache.rowCount = static_cast<int>(getDatasetRowCount(dataset));

    if (newCache.rowCount <= 0) {
        errorMessage = "No hay filas validas para construir la cache GPU.";
        return false;
    }

    std::vector<char> tailBuffer;
    buildTailBuffer(dataset.tailNum, tailBuffer);

    std::vector<int> originDenseInput;
    std::vector<int> destinationDenseInput;

    buildPhase4DenseInput(dataset.originSeqId, dataset.originIdToCode, newCache.origin.denseToSeqId, originDenseInput);
    buildPhase4DenseInput(dataset.destSeqId, dataset.destIdToCode, newCache.destination.denseToSeqId, destinationDenseInput);

    newCache.origin.totalElements = static_cast<int>(originDenseInput.size());
    newCache.origin.totalBins = static_cast<int>(newCache.origin.denseToSeqId.size());
    newCache.destination.totalElements = static_cast<int>(destinationDenseInput.size());
    newCache.destination.totalBins = static_cast<int>(newCache.destination.denseToSeqId.size());

    const std::size_t delayBytes = static_cast<std::size_t>(newCache.rowCount) * sizeof(float);
    const std::size_t tailBytes = tailBuffer.size() * sizeof(char);
    const std::size_t outDelayBytes = static_cast<std::size_t>(newCache.rowCount) * sizeof(int);
    const std::size_t outTailBytes = static_cast<std::size_t>(newCache.rowCount) * kPhase2TailNumStride * sizeof(char);

    auto fail = [&](const char* context, cudaError_t status) {
        errorMessage = std::string(context) + ": " + cudaGetErrorString(status);
        releaseGpuCache(newCache);
        return false;
    };

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&newCache.d_depDelay), delayBytes);
    if (status != cudaSuccess) {
        return fail("cudaMalloc d_depDelay", status);
    }

    status = cudaMalloc(reinterpret_cast<void**>(&newCache.d_arrDelay), delayBytes);
    if (status != cudaSuccess) {
        return fail("cudaMalloc d_arrDelay", status);
    }

    status = cudaMalloc(reinterpret_cast<void**>(&newCache.d_tailNums), tailBytes);
    if (status != cudaSuccess) {
        return fail("cudaMalloc d_tailNums", status);
    }

    status = cudaMalloc(reinterpret_cast<void**>(&newCache.d_phase2Count), sizeof(int));
    if (status != cudaSuccess) {
        return fail("cudaMalloc d_phase2Count", status);
    }

    status = cudaMalloc(reinterpret_cast<void**>(&newCache.d_phase2OutDelayValues), outDelayBytes);
    if (status != cudaSuccess) {
        return fail("cudaMalloc d_phase2OutDelayValues", status);
    }

    status = cudaMalloc(reinterpret_cast<void**>(&newCache.d_phase2OutTailNums), outTailBytes);
    if (status != cudaSuccess) {
        return fail("cudaMalloc d_phase2OutTailNums", status);
    }

    if (newCache.origin.totalElements > 0) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&newCache.origin.d_denseInput),
            static_cast<std::size_t>(newCache.origin.totalElements) * sizeof(int));

        if (status != cudaSuccess) {
            return fail("cudaMalloc origin.d_denseInput", status);
        }
    }

    if (newCache.destination.totalElements > 0) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&newCache.destination.d_denseInput),
            static_cast<std::size_t>(newCache.destination.totalElements) * sizeof(int));

        if (status != cudaSuccess) {
            return fail("cudaMalloc destination.d_denseInput", status);
        }
    }

    status = cudaMemcpy(newCache.d_depDelay, dataset.depDelay.data(), delayBytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return fail("cudaMemcpy H2D d_depDelay", status);
    }

    status = cudaMemcpy(newCache.d_arrDelay, dataset.arrDelay.data(), delayBytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return fail("cudaMemcpy H2D d_arrDelay", status);
    }

    status = cudaMemcpy(newCache.d_tailNums, tailBuffer.data(), tailBytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return fail("cudaMemcpy H2D d_tailNums", status);
    }

    if (newCache.origin.totalElements > 0) {
        status = cudaMemcpy(
            newCache.origin.d_denseInput,
            originDenseInput.data(),
            static_cast<std::size_t>(newCache.origin.totalElements) * sizeof(int),
            cudaMemcpyHostToDevice);

        if (status != cudaSuccess) {
            return fail("cudaMemcpy H2D origin.d_denseInput", status);
        }
    }

    if (newCache.destination.totalElements > 0) {
        status = cudaMemcpy(
            newCache.destination.d_denseInput,
            destinationDenseInput.data(),
            static_cast<std::size_t>(newCache.destination.totalElements) * sizeof(int),
            cudaMemcpyHostToDevice);

        if (status != cudaSuccess) {
            return fail("cudaMemcpy H2D destination.d_denseInput", status);
        }
    }

    releaseGpuCache(cache);
    cache = newCache;
    errorMessage.clear();
    return true;
}

void printLoadSummary(const AppState& appState)
{
    std::cout << "\n=== Fase 0 ===\n";
    std::cout << "Ruta: " << appState.datasetPath << "\n";
    std::cout << "Filas de datos leidas: " << appState.summary.rowsRead << "\n";
    std::cout << "Filas almacenadas: " << appState.summary.storedRows << "\n";
    std::cout << "Filas descartadas: " << appState.summary.discardedRows << "\n";

    std::cout << "\nValores ausentes detectados:\n";
    std::cout << "- TAIL_NUM: " << appState.summary.missingTailNum << "\n";
    std::cout << "- ORIGIN_SEQ_ID: " << appState.summary.missingOriginSeqId << "\n";
    std::cout << "- ORIGIN_AIRPORT: " << appState.summary.missingOriginAirportCode << "\n";
    std::cout << "- DEST_SEQ_ID: " << appState.summary.missingDestSeqId << "\n";
    std::cout << "- DEST_AIRPORT: " << appState.summary.missingDestAirportCode << "\n";
    std::cout << "- DEP_DELAY: " << appState.summary.missingDepDelay << "\n";
    std::cout << "- ARR_DELAY: " << appState.summary.missingArrDelay << "\n";
    std::cout << "- WEATHER_DELAY: " << appState.summary.missingWeatherDelay << "\n";

    std::cout << "\nCategorias detectadas:\n";
    std::cout << "- Aeropuertos unicos de origen por SEQ_ID: "
              << appState.summary.uniqueOriginSeqIds << "\n";
    std::cout << "- Aeropuertos unicos de destino por SEQ_ID: "
              << appState.summary.uniqueDestinationSeqIds << "\n";
}

void printGpuSummary(const AppState& appState)
{
    std::cout << "\n=== CUDA ===\n";

    if (!appState.deviceReady) {
        std::cout << "No disponible: " << appState.deviceErrorMessage << "\n";
        return;
    }

    const unsigned long long globalMemoryInMb =
        static_cast<unsigned long long>(appState.deviceProp.totalGlobalMem) / (1024ULL * 1024ULL);
    const unsigned long long sharedMemoryInKb =
        static_cast<unsigned long long>(appState.deviceProp.sharedMemPerBlock) / 1024ULL;

    std::cout << "GPU: " << appState.deviceProp.name
              << " | CC " << appState.deviceProp.major << "." << appState.deviceProp.minor << "\n";
    std::cout << "Global: " << globalMemoryInMb << " MB"
              << " | Shared por bloque: " << sharedMemoryInKb << " KB"
              << " | Max hilos/bloque: " << appState.deviceProp.maxThreadsPerBlock << "\n";

    if (appState.datasetLoaded) {
        const int rowCount = static_cast<int>(getDatasetRowCount(appState.dataset));
        const LaunchConfig launchConfig = computeLaunchConfig(rowCount, appState.deviceProp);

        std::cout << "Sugerencia base: " << launchConfig.blocks
                  << " bloques x " << launchConfig.threadsPerBlock << " hilos\n";

        if (appState.gpu.rowCount > 0) {
            std::cout << "Cache GPU lista para Fases 01, 02 y 04.\n";
        }
    }
}

void printApplicationState(const AppState& appState)
{
    if (appState.datasetLoaded) {
        printLoadSummary(appState);
    } else {
        std::cout << "\nDataset no cargado.\n";
    }

    printGpuSummary(appState);
}

bool loadDatasetIntoState(AppState& appState, const std::string& datasetPath)
{
    DatasetColumns newDataset;
    LoadSummary newSummary;
    std::string errorMessage;

    if (!loadDataset(datasetPath, newDataset, newSummary, errorMessage)) {
        std::cout << "No se ha podido cargar el dataset.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    GpuCache newGpuCache;

    if (appState.deviceReady && !buildGpuCache(newDataset, newGpuCache, errorMessage)) {
        std::cout << "No se ha podido construir la cache GPU.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    releaseGpuCache(appState.gpu);
    appState.datasetPath = datasetPath;
    appState.dataset = std::move(newDataset);
    appState.summary = newSummary;
    appState.gpu = newGpuCache;
    appState.datasetLoaded = true;

    printLoadSummary(appState);
    printGpuSummary(appState);
    return true;
}

bool promptAndLoadDataset(AppState& appState, bool allowCancel)
{
    while (true) {
        std::vector<std::string> candidates;
        std::string defaultPath;

        if (!appState.datasetPath.empty()) {
            candidates.push_back(appState.datasetPath);
        }

        candidates.push_back("src/data/Airline_dataset.csv");

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (!candidates[i].empty() && fileExists(candidates[i])) {
                defaultPath = candidates[i];
                break;
            }
        }

        std::cout << "\nRuta del CSV";

        if (!defaultPath.empty()) {
            std::cout << " [Intro = " << defaultPath << "]";
        }

        std::cout << " [X = volver]\n";
        std::cout << "> ";

        std::string selectedPath;
        std::getline(std::cin, selectedPath);
        selectedPath = trimWhitespace(selectedPath);

        if (isCancelToken(selectedPath)) {
            if (allowCancel) {
                return false;
            }

            std::cout << "Debe indicar una ruta valida.\n";
            continue;
        }

        if (selectedPath.empty()) {
            selectedPath = defaultPath;
        }

        if (selectedPath.empty()) {
            std::cout << "Debe indicar una ruta valida.\n";
            continue;
        }

        if (loadDatasetIntoState(appState, selectedPath)) {
            return true;
        }
    }
}

bool canRunGpuPhase(const AppState& appState)
{
    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado.\n";
        return false;
    }

    if (!appState.deviceReady) {
        std::cout << "No hay GPU CUDA disponible.\n";
        std::cout << "Motivo: " << appState.deviceErrorMessage << "\n";
        return false;
    }

    if (appState.gpu.rowCount <= 0) {
        std::cout << "La cache GPU no esta disponible.\n";
        return false;
    }

    return true;
}

void printPhase2HostSummary(
    DelayFilterMode mode,
    int threshold,
    int resultCount,
    const std::vector<int>& outDelayValues,
    const std::vector<char>& outTailNumBuffer)
{
    std::cout << "\nResultados CPU:\n";
    std::cout << "Se han encontrado " << resultCount << " aviones\n";

    for (int i = 0; i < resultCount; ++i) {
        const char* tailNum = &outTailNumBuffer[static_cast<std::size_t>(i) * kPhase2TailNumStride];
        const int detectedValue = outDelayValues[static_cast<std::size_t>(i)];

        std::cout << "- Matricula " << tailNum
                  << " | " << getDetectedDelayLabel(detectedValue, mode, threshold)
                  << ": " << detectedValue << " minutos\n";
    }
}

bool runPhase1Computation(const AppState& appState, DelayFilterMode mode, int threshold)
{
    const int totalElements = appState.gpu.rowCount;
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    std::cout << "DEP_DELAY | modo " << getDelayModeLabel(mode)
              << " | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    phase1DepartureDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        appState.gpu.d_depDelay,
        totalElements,
        static_cast<int>(mode),
        threshold);

    if (!cudaOk(cudaGetLastError(), "lanzamiento phase1DepartureDelayKernel")) {
        return false;
    }

    return cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize Fase 01");
}

bool runPhase2Computation(const AppState& appState, DelayFilterMode mode, int threshold)
{
    const int totalElements = appState.gpu.rowCount;
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    std::cout << "ARR_DELAY + TAIL_NUM | modo " << getDelayModeLabel(mode)
              << " | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    if (!cudaOk(cudaMemset(appState.gpu.d_phase2Count, 0, sizeof(int)), "cudaMemset d_phase2Count")) {
        return false;
    }

    if (!cudaOk(copyPhase2FilterConfigToConstant(static_cast<int>(mode), threshold), "copyPhase2FilterConfigToConstant")) {
        return false;
    }

    phase2ArrivalDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        appState.gpu.d_arrDelay,
        appState.gpu.d_tailNums,
        totalElements,
        appState.gpu.d_phase2Count,
        appState.gpu.d_phase2OutDelayValues,
        appState.gpu.d_phase2OutTailNums);

    if (!cudaOk(cudaGetLastError(), "lanzamiento phase2ArrivalDelayKernel")) {
        return false;
    }

    if (!cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize Fase 02")) {
        return false;
    }

    int resultCount = 0;

    if (!cudaOk(
            cudaMemcpy(&resultCount, appState.gpu.d_phase2Count, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H d_phase2Count")) {
        return false;
    }

    std::vector<int> outDelayValues(static_cast<std::size_t>(resultCount));
    std::vector<char> outTailNumBuffer(static_cast<std::size_t>(resultCount) * kPhase2TailNumStride, '\0');

    if (resultCount > 0) {
        if (!cudaOk(
                cudaMemcpy(
                    outDelayValues.data(),
                    appState.gpu.d_phase2OutDelayValues,
                    static_cast<std::size_t>(resultCount) * sizeof(int),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H d_phase2OutDelayValues")) {
            return false;
        }

        if (!cudaOk(
                cudaMemcpy(
                    outTailNumBuffer.data(),
                    appState.gpu.d_phase2OutTailNums,
                    static_cast<std::size_t>(resultCount) * kPhase2TailNumStride * sizeof(char),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H d_phase2OutTailNums")) {
            return false;
        }
    }

    printPhase2HostSummary(mode, threshold, resultCount, outDelayValues, outTailNumBuffer);
    return true;
}

bool runPhase3AtomicVariant(
    Phase3AtomicVariant variant,
    int* deviceInput,
    int totalElements,
    bool isMax,
    const LaunchConfig& launchConfig,
    int& outResult)
{
    int* deviceResult = nullptr;
    const int initialValue = getReductionIdentity(isMax);
    std::size_t sharedBytes = 0;

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(int)), "cudaMalloc deviceResult")) {
        return false;
    }

    if (!cudaOk(
            cudaMemcpy(deviceResult, &initialValue, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D deviceResult")) {
        cudaFree(deviceResult);
        return false;
    }

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

    if (!cudaOk(cudaGetLastError(), "lanzamiento Fase 03 atomica")) {
        cudaFree(deviceResult);
        return false;
    }

    if (!cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize Fase 03 atomica")) {
        cudaFree(deviceResult);
        return false;
    }

    if (!cudaOk(
            cudaMemcpy(&outResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H deviceResult")) {
        cudaFree(deviceResult);
        return false;
    }

    cudaFree(deviceResult);
    return true;
}

bool runPhase3ReductionVariant(
    int* deviceInput,
    int totalElements,
    bool isMax,
    const cudaDeviceProp& deviceProp,
    int& outResult)
{
    int* currentInput = deviceInput;
    bool ownsCurrentInput = false;
    int currentCount = totalElements;

    while (currentCount > 10) {
        const LaunchConfig launchConfig = computeLaunchConfig(currentCount, deviceProp);
        const std::size_t partialBytes = static_cast<std::size_t>(launchConfig.blocks) * sizeof(int);
        const std::size_t sharedBytes = static_cast<std::size_t>(launchConfig.threadsPerBlock) * sizeof(int);

        int* devicePartials = nullptr;

        if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&devicePartials), partialBytes), "cudaMalloc devicePartials")) {
            if (ownsCurrentInput) {
                cudaFree(currentInput);
            }
            return false;
        }

        reductionPattern<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
            currentInput,
            devicePartials,
            currentCount,
            isMax);

        if (!cudaOk(cudaGetLastError(), "lanzamiento reductionPattern")) {
            cudaFree(devicePartials);
            if (ownsCurrentInput) {
                cudaFree(currentInput);
            }
            return false;
        }

        if (!cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize reductionPattern")) {
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

    if (!cudaOk(
            cudaMemcpy(
                hostFinalValues.data(),
                currentInput,
                static_cast<std::size_t>(currentCount) * sizeof(int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H vector final Fase 03")) {
        if (ownsCurrentInput) {
            cudaFree(currentInput);
        }
        return false;
    }

    outResult = hostFinalValues[0];

    for (int i = 1; i < currentCount; ++i) {
        outResult = hostCompareReduction(outResult, hostFinalValues[static_cast<std::size_t>(i)], isMax);
    }

    if (ownsCurrentInput) {
        cudaFree(currentInput);
    }

    return true;
}

bool runPhase3Computation(
    const AppState& appState,
    Phase3ColumnOption columnOption,
    ReductionTypeOption reductionOption)
{
    const std::vector<float>* sourceColumn = nullptr;

    switch (columnOption) {
    case Phase3ColumnOption::ArrivalDelay:
        sourceColumn = &appState.dataset.arrDelay;
        break;
    case Phase3ColumnOption::WeatherDelay:
        sourceColumn = &appState.dataset.weatherDelay;
        break;
    case Phase3ColumnOption::DepartureDelay:
    default:
        sourceColumn = &appState.dataset.depDelay;
        break;
    }

    std::vector<int> inputValues;
    inputValues.reserve(sourceColumn->size());

    for (std::size_t i = 0; i < sourceColumn->size(); ++i) {
        const float value = (*sourceColumn)[i];

        if (std::isnan(value)) {
            continue;
        }

        inputValues.push_back(static_cast<int>(value));
    }

    const int totalElements = static_cast<int>(inputValues.size());

    if (totalElements <= 0) {
        std::cout << "No hay valores validos para la Fase 03.\n";
        return false;
    }

    const bool isMax = reductionOption == ReductionTypeOption::Maximum;
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    std::cout << getPhase3ColumnLabel(columnOption)
              << " | " << getReductionLabel(reductionOption)
              << " | validos " << totalElements
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    int* deviceInput = nullptr;

    if (!cudaOk(
            cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputValues.size() * sizeof(int)),
            "cudaMalloc deviceInput Fase 03")) {
        return false;
    }

    if (!cudaOk(
            cudaMemcpy(
                deviceInput,
                inputValues.data(),
                inputValues.size() * sizeof(int),
                cudaMemcpyHostToDevice),
            "cudaMemcpy H2D deviceInput Fase 03")) {
        cudaFree(deviceInput);
        return false;
    }

    int simpleResult = 0;
    int basicResult = 0;
    int intermediateResult = 0;
    int reductionResult = 0;

    if (!runPhase3AtomicVariant(
            Phase3AtomicVariant::Simple,
            deviceInput,
            totalElements,
            isMax,
            launchConfig,
            simpleResult) ||
        !runPhase3AtomicVariant(
            Phase3AtomicVariant::Basic,
            deviceInput,
            totalElements,
            isMax,
            launchConfig,
            basicResult) ||
        !runPhase3AtomicVariant(
            Phase3AtomicVariant::Intermediate,
            deviceInput,
            totalElements,
            isMax,
            launchConfig,
            intermediateResult) ||
        !runPhase3ReductionVariant(deviceInput, totalElements, isMax, appState.deviceProp, reductionResult)) {
        cudaFree(deviceInput);
        return false;
    }

    cudaFree(deviceInput);

    std::cout << "\n[Simple] " << getReductionFunctionLabel(reductionOption)
              << "() " << getPhase3ColumnLabel(columnOption)
              << " = " << simpleResult << " minutos\n";
    std::cout << "[Basica] " << getReductionFunctionLabel(reductionOption)
              << "() " << getPhase3ColumnLabel(columnOption)
              << " = " << basicResult << " minutos\n";
    std::cout << "[Intermedia] " << getReductionFunctionLabel(reductionOption)
              << "() " << getPhase3ColumnLabel(columnOption)
              << " = " << intermediateResult << " minutos\n";
    std::cout << "[Reduccion] " << getReductionFunctionLabel(reductionOption)
              << "() " << getPhase3ColumnLabel(columnOption)
              << " = " << reductionResult << " minutos\n";

    return true;
}

void printPhase4Histogram(
    HistogramAirportTypeOption airportType,
    int threshold,
    const std::vector<unsigned int>& histogram,
    const std::vector<int>& denseToSeqId,
    const std::unordered_map<int, std::string>& idToCode)
{
    const unsigned int minimumCount = static_cast<unsigned int>(threshold);
    unsigned int maximumShownCount = 0;
    int shownAirports = 0;

    for (std::size_t i = 0; i < histogram.size(); ++i) {
        if (histogram[i] >= minimumCount) {
            ++shownAirports;

            if (histogram[i] > maximumShownCount) {
                maximumShownCount = histogram[i];
            }
        }
    }

    std::cout << "\n(4) Histograma de aeropuertos de " << getAirportLabel(airportType) << "\n";
    std::cout << "Num de aeropuertos encontrados: " << denseToSeqId.size() << "\n\n";

    for (std::size_t denseIndex = 0; denseIndex < histogram.size(); ++denseIndex) {
        const unsigned int airportCount = histogram[denseIndex];

        if (airportCount < minimumCount) {
            continue;
        }

        const int seqId = denseToSeqId[denseIndex];
        std::unordered_map<int, std::string>::const_iterator codeIt = idToCode.find(seqId);
        const std::string airportCode = codeIt == idToCode.end() ? "" : codeIt->second;

        std::cout << airportCode << " (" << seqId << ") | " << airportCount << " ";

        int barLength = 0;
        const int maxBarWidth = 40;

        if (maximumShownCount > 0) {
            barLength = static_cast<int>(
                (static_cast<unsigned long long>(airportCount) * static_cast<unsigned long long>(maxBarWidth)) /
                static_cast<unsigned long long>(maximumShownCount));
        }

        if (barLength <= 0 && airportCount > 0) {
            barLength = 1;
        }

        for (int i = 0; i < barLength; ++i) {
            std::cout << '#';
        }

        std::cout << "\n";
    }

    std::cout << "\nAeropuertos mostrados (con al menos " << threshold
              << " vuelos): " << shownAirports
              << " (del total " << denseToSeqId.size() << ")\n";
}

bool runPhase4Computation(const AppState& appState, HistogramAirportTypeOption airportType, int threshold)
{
    const Phase4Cache& cache =
        airportType == HistogramAirportTypeOption::Origin ? appState.gpu.origin : appState.gpu.destination;
    const std::unordered_map<int, std::string>& idToCode =
        airportType == HistogramAirportTypeOption::Origin ? appState.dataset.originIdToCode : appState.dataset.destIdToCode;

    if (cache.totalElements <= 0 || cache.totalBins <= 0 || cache.d_denseInput == nullptr) {
        std::cout << "No hay datos validos para la Fase 04.\n";
        return false;
    }

    const std::size_t sharedBytes = static_cast<std::size_t>(cache.totalBins) * sizeof(unsigned int);

    if (sharedBytes > static_cast<std::size_t>(appState.deviceProp.sharedMemPerBlock)) {
        std::cout << "El histograma no cabe en la memoria compartida por bloque de esta GPU.\n";
        return false;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(cache.totalElements, appState.deviceProp);

    std::cout << getAirportLabel(airportType)
              << " | filas validas " << cache.totalElements
              << " | bins " << cache.totalBins
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    unsigned int* devicePartialHistograms = nullptr;
    unsigned int* deviceFinalHistogram = nullptr;

    const std::size_t partialBytes =
        static_cast<std::size_t>(launchConfig.blocks) * static_cast<std::size_t>(cache.totalBins) * sizeof(unsigned int);
    const std::size_t finalBytes = static_cast<std::size_t>(cache.totalBins) * sizeof(unsigned int);

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&devicePartialHistograms), partialBytes), "cudaMalloc devicePartialHistograms")) {
        return false;
    }

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&deviceFinalHistogram), finalBytes), "cudaMalloc deviceFinalHistogram")) {
        cudaFree(devicePartialHistograms);
        return false;
    }

    phase4SharedHistogramKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
        cache.d_denseInput,
        cache.totalElements,
        cache.totalBins,
        devicePartialHistograms);

    if (!cudaOk(cudaGetLastError(), "lanzamiento phase4SharedHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    if (!cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize phase4SharedHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    const LaunchConfig mergeLaunchConfig = computeLaunchConfig(cache.totalBins, appState.deviceProp);

    phase4MergeHistogramKernel<<<mergeLaunchConfig.blocks, mergeLaunchConfig.threadsPerBlock>>>(
        devicePartialHistograms,
        launchConfig.blocks,
        cache.totalBins,
        deviceFinalHistogram);

    if (!cudaOk(cudaGetLastError(), "lanzamiento phase4MergeHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    if (!cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize phase4MergeHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    std::vector<unsigned int> histogram(static_cast<std::size_t>(cache.totalBins), 0U);

    if (!cudaOk(
            cudaMemcpy(histogram.data(), deviceFinalHistogram, finalBytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H deviceFinalHistogram")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    cudaFree(devicePartialHistograms);
    cudaFree(deviceFinalHistogram);

    printPhase4Histogram(airportType, threshold, histogram, cache.denseToSeqId, idToCode);
    return true;
}

} // namespace

int main()
{
    AppState appState;

    std::cout << "========================================\n";
    std::cout << " PL1 CUDA - US Airline Dataset Toolkit\n";
    std::cout << "========================================\n";

    appState.deviceReady = queryGpuInfo(appState.deviceProp, appState.deviceErrorMessage);
    printGpuSummary(appState);

    if (!promptAndLoadDataset(appState, true)) {
        std::cout << "Saliendo sin cargar dataset.\n";
        return 0;
    }

    std::cout << "\nPulse Intro para continuar...";
    {
        std::string dummy;
        std::getline(std::cin, dummy);
    }

    bool keepRunning = true;

    while (keepRunning) {
        MainMenuOption selectedOption = MainMenuOption::Exit;

        while (true) {
            std::cout << "\nMenu principal\n";
            std::cout << "1. Fase 01 - Retraso en salida\n";
            std::cout << "2. Fase 02 - Retraso en llegada\n";
            std::cout << "3. Fase 03 - Reduccion de retraso\n";
            std::cout << "4. Fase 04 - Histograma de aeropuertos\n";
            std::cout << "R. Recargar CSV\n";
            std::cout << "I. Ver estado de la aplicacion\n";
            std::cout << "X. Salir\n";
            std::cout << "> ";

            std::string optionInput;
            std::getline(std::cin, optionInput);
            optionInput = trimWhitespace(optionInput);

            if (optionInput == "1") {
                selectedOption = MainMenuOption::Phase1;
                break;
            }

            if (optionInput == "2") {
                selectedOption = MainMenuOption::Phase2;
                break;
            }

            if (optionInput == "3") {
                selectedOption = MainMenuOption::Phase3;
                break;
            }

            if (optionInput == "4") {
                selectedOption = MainMenuOption::Phase4;
                break;
            }

            if (optionInput == "R" || optionInput == "r") {
                selectedOption = MainMenuOption::ReloadCsv;
                break;
            }

            if (optionInput == "I" || optionInput == "i") {
                selectedOption = MainMenuOption::ShowStatus;
                break;
            }

            if (isCancelToken(optionInput)) {
                selectedOption = MainMenuOption::Exit;
                break;
            }

            std::cout << "Opcion no valida.\n";
        }

        switch (selectedOption) {
        case MainMenuOption::Phase1:
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 01 - DEP_DELAY\n";

            {
                DelayFilterMode mode = DelayFilterMode::Both;
                int threshold = 0;
                bool cancelled = false;

                while (true) {
                    std::cout << "1. Retraso  2. Adelanto  3. Ambos  Intro. Ambos\n";
                    std::cout << "Modo: ";

                    std::string modeInput;
                    std::getline(std::cin, modeInput);
                    modeInput = trimWhitespace(modeInput);

                    if (isCancelToken(modeInput)) {
                        cancelled = true;
                        break;
                    }

                    if (modeInput.empty() || modeInput == "3") {
                        mode = DelayFilterMode::Both;
                        break;
                    }

                    if (modeInput == "1") {
                        mode = DelayFilterMode::Delay;
                        break;
                    }

                    if (modeInput == "2") {
                        mode = DelayFilterMode::Advance;
                        break;
                    }

                    std::cout << "Debe introducir 1, 2, 3, Intro o X.\n";
                }

                while (!cancelled) {
                    std::cout << "Umbral (>= 0, X para volver): ";

                    std::string thresholdInput;
                    std::getline(std::cin, thresholdInput);
                    thresholdInput = trimWhitespace(thresholdInput);

                    if (isCancelToken(thresholdInput)) {
                        cancelled = true;
                        break;
                    }

                    std::stringstream parser(thresholdInput);
                    int parsedValue = 0;
                    char trailingCharacter = '\0';

                    if ((parser >> parsedValue) &&
                        !(parser >> trailingCharacter) &&
                        parsedValue >= 0) {
                        threshold = parsedValue;
                        break;
                    }

                    std::cout << "Debe introducir un numero mayor o igual que 0, o X.\n";
                }

                if (!cancelled) {
                    if (!runPhase1Computation(appState, mode, threshold)) {
                        std::cout << "La Fase 01 no se ha podido completar.\n";
                    }

                    std::cout << "\nPulse Intro para continuar...";
                    std::string dummy;
                    std::getline(std::cin, dummy);
                }
            }
            break;

        case MainMenuOption::Phase2:
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 02 - ARR_DELAY + TAIL_NUM\n";

            {
                DelayFilterMode mode = DelayFilterMode::Both;
                int threshold = 0;
                bool cancelled = false;

                while (true) {
                    std::cout << "1. Retraso  2. Adelanto  3. Ambos  Intro. Ambos\n";
                    std::cout << "Modo: ";

                    std::string modeInput;
                    std::getline(std::cin, modeInput);
                    modeInput = trimWhitespace(modeInput);

                    if (isCancelToken(modeInput)) {
                        cancelled = true;
                        break;
                    }

                    if (modeInput.empty() || modeInput == "3") {
                        mode = DelayFilterMode::Both;
                        break;
                    }

                    if (modeInput == "1") {
                        mode = DelayFilterMode::Delay;
                        break;
                    }

                    if (modeInput == "2") {
                        mode = DelayFilterMode::Advance;
                        break;
                    }

                    std::cout << "Debe introducir 1, 2, 3, Intro o X.\n";
                }

                while (!cancelled) {
                    std::cout << "Umbral (>= 0, X para volver): ";

                    std::string thresholdInput;
                    std::getline(std::cin, thresholdInput);
                    thresholdInput = trimWhitespace(thresholdInput);

                    if (isCancelToken(thresholdInput)) {
                        cancelled = true;
                        break;
                    }

                    std::stringstream parser(thresholdInput);
                    int parsedValue = 0;
                    char trailingCharacter = '\0';

                    if ((parser >> parsedValue) &&
                        !(parser >> trailingCharacter) &&
                        parsedValue >= 0) {
                        threshold = parsedValue;
                        break;
                    }

                    std::cout << "Debe introducir un numero mayor o igual que 0, o X.\n";
                }

                if (!cancelled) {
                    if (!runPhase2Computation(appState, mode, threshold)) {
                        std::cout << "La Fase 02 no se ha podido completar.\n";
                    }

                    std::cout << "\nPulse Intro para continuar...";
                    std::string dummy;
                    std::getline(std::cin, dummy);
                }
            }
            break;

        case MainMenuOption::Phase3:
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 03 - Reduccion\n";
            std::cout << "1. DEP_DELAY  2. ARR_DELAY  3. WEATHER_DELAY\n";

            {
                int columnValue = 0;
                int reductionValue = 0;
                bool cancelled = false;

                while (true) {
                    std::cout << "Columna: ";

                    std::string columnInput;
                    std::getline(std::cin, columnInput);
                    columnInput = trimWhitespace(columnInput);

                    if (isCancelToken(columnInput)) {
                        cancelled = true;
                        break;
                    }

                    std::stringstream parser(columnInput);
                    int parsedValue = 0;
                    char trailingCharacter = '\0';

                    if ((parser >> parsedValue) &&
                        !(parser >> trailingCharacter) &&
                        parsedValue >= 1 &&
                        parsedValue <= 3) {
                        columnValue = parsedValue;
                        break;
                    }

                    std::cout << "Debe introducir un numero entre 1 y 3, o X.\n";
                }

                if (!cancelled) {
                    std::cout << "1. Maximo  2. Minimo\n";

                    while (true) {
                        std::cout << "Reduccion: ";

                        std::string reductionInput;
                        std::getline(std::cin, reductionInput);
                        reductionInput = trimWhitespace(reductionInput);

                        if (isCancelToken(reductionInput)) {
                            cancelled = true;
                            break;
                        }

                        std::stringstream parser(reductionInput);
                        int parsedValue = 0;
                        char trailingCharacter = '\0';

                        if ((parser >> parsedValue) &&
                            !(parser >> trailingCharacter) &&
                            parsedValue >= 1 &&
                            parsedValue <= 2) {
                            reductionValue = parsedValue;
                            break;
                        }

                        std::cout << "Debe introducir un numero entre 1 y 2, o X.\n";
                    }
                }

                if (!cancelled) {
                    if (!runPhase3Computation(
                            appState,
                            static_cast<Phase3ColumnOption>(columnValue),
                            static_cast<ReductionTypeOption>(reductionValue))) {
                        std::cout << "La Fase 03 no se ha podido completar.\n";
                    }

                    std::cout << "\nPulse Intro para continuar...";
                    std::string dummy;
                    std::getline(std::cin, dummy);
                }
            }
            break;

        case MainMenuOption::Phase4:
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 04 - Histograma de aeropuertos\n";
            std::cout << "1. Origen  2. Destino\n";

            {
                int airportValue = 0;
                int threshold = 0;
                bool cancelled = false;

                while (true) {
                    std::cout << "Tipo de aeropuerto: ";

                    std::string airportInput;
                    std::getline(std::cin, airportInput);
                    airportInput = trimWhitespace(airportInput);

                    if (isCancelToken(airportInput)) {
                        cancelled = true;
                        break;
                    }

                    std::stringstream parser(airportInput);
                    int parsedValue = 0;
                    char trailingCharacter = '\0';

                    if ((parser >> parsedValue) &&
                        !(parser >> trailingCharacter) &&
                        parsedValue >= 1 &&
                        parsedValue <= 2) {
                        airportValue = parsedValue;
                        break;
                    }

                    std::cout << "Debe introducir un numero entre 1 y 2, o X.\n";
                }

                while (!cancelled) {
                    std::cout << "Umbral minimo (>= 0, X para volver): ";

                    std::string thresholdInput;
                    std::getline(std::cin, thresholdInput);
                    thresholdInput = trimWhitespace(thresholdInput);

                    if (isCancelToken(thresholdInput)) {
                        cancelled = true;
                        break;
                    }

                    std::stringstream parser(thresholdInput);
                    int parsedValue = 0;
                    char trailingCharacter = '\0';

                    if ((parser >> parsedValue) &&
                        !(parser >> trailingCharacter) &&
                        parsedValue >= 0) {
                        threshold = parsedValue;
                        break;
                    }

                    std::cout << "Debe introducir un numero mayor o igual que 0, o X.\n";
                }

                if (!cancelled) {
                    if (!runPhase4Computation(
                            appState,
                            static_cast<HistogramAirportTypeOption>(airportValue),
                            threshold)) {
                        std::cout << "La Fase 04 no se ha podido completar.\n";
                    }

                    std::cout << "\nPulse Intro para continuar...";
                    std::string dummy;
                    std::getline(std::cin, dummy);
                }
            }
            break;

        case MainMenuOption::ReloadCsv:
            promptAndLoadDataset(appState, true);
            break;

        case MainMenuOption::ShowStatus:
            printApplicationState(appState);
            std::cout << "\nPulse Intro para continuar...";
            {
                std::string dummy;
                std::getline(std::cin, dummy);
            }
            break;

        case MainMenuOption::Exit:
            keepRunning = false;
            break;
        }
    }

    releaseGpuCache(appState.gpu);
    std::cout << "\nAplicacion finalizada.\n";
    return 0;
}
