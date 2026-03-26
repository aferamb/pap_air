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

    Esta version compacta el host en torno a tres ideas:

    - una CLI centralizada en este archivo, pero con helpers pequenos;
    - un dataset persistente en GPU que se construye una sola vez al cargar;
    - menos pegamento repetido alrededor de los lanzamientos CUDA.

    Se mantiene la funcionalidad importante de la practica:

    - Fase 01 sobre DEP_DELAY;
    - Fase 02 sobre ARR_DELAY + TAIL_NUM con memoria constante y atomicas;
    - Fase 03 completa con sus cuatro variantes y WEATHER_DELAY;
    - Fase 04 con bins densos por SEQ_ID y memoria compartida.
*/

#define CUDA_RETURN_FALSE(call) \
    do { \
        if (!cudaOk((call), #call)) return false; \
    } while (0)

/*
    Opciones de menu y tipos pequenos de seleccion.

    Se mantienen como enums simples para que el flujo del programa siga siendo
    legible sin recurrir a cadenas magicas.
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

/*
    LaunchConfig

    Estructura minima para describir el lanzamiento 1D que se muestra y se usa
    en las fases CUDA.
*/
struct LaunchConfig {
    int blocks = 0;
    int threadsPerBlock = 1;
};

/*
    DeviceDataset

    Representa el dataset persistente en GPU. La idea es guardar aqui solo la
    informacion que se reutiliza de verdad muchas veces:

    - DEP_DELAY para Fase 01;
    - ARR_DELAY y TAIL_NUM para Fase 02;
    - bins densos de origen y destino para Fase 04;
    - buffers de salida persistentes de Fase 02.

    WEATHER_DELAY no se sube aqui de forma fija porque Fase 03 sigue
    construyendo su vector compacto segun la columna elegida.
*/
struct DeviceDataset {
    int rowCount = 0;

    float* d_depDelay = nullptr;
    float* d_arrDelay = nullptr;
    char* d_tailNums = nullptr;

    int* d_phase2Count = nullptr;
    int* d_phase2OutDelayValues = nullptr;
    char* d_phase2OutTailNums = nullptr;

    int* d_originDenseInput = nullptr;
    int originTotalElements = 0;
    int originTotalBins = 0;
    std::vector<int> originDenseToSeqId;

    int* d_destinationDenseInput = nullptr;
    int destinationTotalElements = 0;
    int destinationTotalBins = 0;
    std::vector<int> destinationDenseToSeqId;
};

/*
    AppState

    Estado vivo de la aplicacion:

    - ruta activa del CSV;
    - dataset limpio en host;
    - resumen de la Fase 0;
    - informacion CUDA;
    - copia persistente del dataset util en GPU.
*/
struct AppState {
    std::string datasetPath;
    bool datasetLoaded = false;
    DatasetColumns dataset;
    LoadSummary summary;

    bool deviceReady = false;
    cudaDeviceProp deviceProp{};
    std::string deviceErrorMessage;
    DeviceDataset device;
};

namespace {

enum class Phase3AtomicVariant {
    Simple,
    Basic,
    Intermediate
};

/*
    Helpers genericos pequenos
*/

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

bool finishKernel(const char* context)
{
    if (!cudaOk(cudaGetLastError(), context)) {
        return false;
    }

    return cudaOk(cudaDeviceSynchronize(), context);
}

void pauseForEnter()
{
    std::cout << "\nPulse Intro para continuar...";
    std::string dummy;
    std::getline(std::cin, dummy);
}

bool readIntegerInRange(const char* prompt, const char* errorMessage, int minValue, int maxValue, int& value)
{
    while (true) {
        std::cout << prompt;

        std::string input;
        std::getline(std::cin, input);
        input = trimWhitespace(input);

        if (isCancelToken(input)) {
            return false;
        }

        std::stringstream parser(input);
        int parsedValue = 0;
        char trailingCharacter = '\0';

        if ((parser >> parsedValue) &&
            !(parser >> trailingCharacter) &&
            parsedValue >= minValue &&
            parsedValue <= maxValue) {
            value = parsedValue;
            return true;
        }

        std::cout << errorMessage << "\n";
    }
}

bool readSignedThreshold(const char* prompt, int& value)
{
    while (true) {
        std::cout << prompt;

        std::string input;
        std::getline(std::cin, input);
        input = trimWhitespace(input);

        if (isCancelToken(input)) {
            return false;
        }

        std::stringstream parser(input);
        int parsedValue = 0;
        char trailingCharacter = '\0';

        if ((parser >> parsedValue) && !(parser >> trailingCharacter)) {
            value = parsedValue;
            return true;
        }

        std::cout << "Debe introducir un numero entero, o X.\n";
    }
}

/*
    Helpers de configuracion y comparacion
*/

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

const std::vector<float>& selectPhase3Column(const DatasetColumns& dataset, Phase3ColumnOption columnOption)
{
    switch (columnOption) {
    case Phase3ColumnOption::ArrivalDelay:
        return dataset.arrDelay;
    case Phase3ColumnOption::WeatherDelay:
        return dataset.weatherDelay;
    case Phase3ColumnOption::DepartureDelay:
    default:
        return dataset.depDelay;
    }
}

/*
    Construccion y liberacion del dataset persistente en GPU
*/

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

void buildDenseInput(
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

        if (seqId < 0 || idToCode.find(seqId) == idToCode.end()) {
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

void releaseDeviceDataset(DeviceDataset& device)
{
    auto freeIfNeeded = [](void* ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    };

    freeIfNeeded(device.d_depDelay);
    freeIfNeeded(device.d_arrDelay);
    freeIfNeeded(device.d_tailNums);
    freeIfNeeded(device.d_phase2Count);
    freeIfNeeded(device.d_phase2OutDelayValues);
    freeIfNeeded(device.d_phase2OutTailNums);
    freeIfNeeded(device.d_originDenseInput);
    freeIfNeeded(device.d_destinationDenseInput);

    device = DeviceDataset{};
}

bool buildDeviceDataset(const DatasetColumns& dataset, DeviceDataset& device, std::string& errorMessage)
{
    DeviceDataset newDevice;
    newDevice.rowCount = static_cast<int>(getDatasetRowCount(dataset));

    if (newDevice.rowCount <= 0) {
        errorMessage = "No hay filas validas para construir el dataset persistente en GPU.";
        return false;
    }

    std::vector<char> tailBuffer;
    std::vector<int> originDenseInput;
    std::vector<int> destinationDenseInput;

    buildTailBuffer(dataset.tailNum, tailBuffer);
    buildDenseInput(dataset.originSeqId, dataset.originIdToCode, newDevice.originDenseToSeqId, originDenseInput);
    buildDenseInput(dataset.destSeqId, dataset.destIdToCode, newDevice.destinationDenseToSeqId, destinationDenseInput);

    newDevice.originTotalElements = static_cast<int>(originDenseInput.size());
    newDevice.originTotalBins = static_cast<int>(newDevice.originDenseToSeqId.size());
    newDevice.destinationTotalElements = static_cast<int>(destinationDenseInput.size());
    newDevice.destinationTotalBins = static_cast<int>(newDevice.destinationDenseToSeqId.size());

    const std::size_t delayBytes = static_cast<std::size_t>(newDevice.rowCount) * sizeof(float);
    const std::size_t tailBytes = tailBuffer.size() * sizeof(char);
    const std::size_t outDelayBytes = static_cast<std::size_t>(newDevice.rowCount) * sizeof(int);
    const std::size_t outTailBytes = static_cast<std::size_t>(newDevice.rowCount) * kPhase2TailNumStride * sizeof(char);

    auto fail = [&](const char* context, cudaError_t status) {
        errorMessage = std::string(context) + ": " + cudaGetErrorString(status);
        releaseDeviceDataset(newDevice);
        return false;
    };

    auto allocate = [&](void** pointer, std::size_t bytes, const char* context) {
        const cudaError_t status = cudaMalloc(pointer, bytes);

        if (status != cudaSuccess) {
            return fail(context, status);
        }

        return true;
    };

    auto upload = [&](void* destination, const void* source, std::size_t bytes, const char* context) {
        const cudaError_t status = cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);

        if (status != cudaSuccess) {
            return fail(context, status);
        }

        return true;
    };

    if (!allocate(reinterpret_cast<void**>(&newDevice.d_depDelay), delayBytes, "cudaMalloc d_depDelay")) {
        return false;
    }

    if (!allocate(reinterpret_cast<void**>(&newDevice.d_arrDelay), delayBytes, "cudaMalloc d_arrDelay")) {
        return false;
    }

    if (!allocate(reinterpret_cast<void**>(&newDevice.d_tailNums), tailBytes, "cudaMalloc d_tailNums")) {
        return false;
    }

    if (!allocate(reinterpret_cast<void**>(&newDevice.d_phase2Count), sizeof(int), "cudaMalloc d_phase2Count")) {
        return false;
    }

    if (!allocate(
            reinterpret_cast<void**>(&newDevice.d_phase2OutDelayValues),
            outDelayBytes,
            "cudaMalloc d_phase2OutDelayValues")) {
        return false;
    }

    if (!allocate(
            reinterpret_cast<void**>(&newDevice.d_phase2OutTailNums),
            outTailBytes,
            "cudaMalloc d_phase2OutTailNums")) {
        return false;
    }

    if (newDevice.originTotalElements > 0 &&
        !allocate(
            reinterpret_cast<void**>(&newDevice.d_originDenseInput),
            static_cast<std::size_t>(newDevice.originTotalElements) * sizeof(int),
            "cudaMalloc d_originDenseInput")) {
        return false;
    }

    if (newDevice.destinationTotalElements > 0 &&
        !allocate(
            reinterpret_cast<void**>(&newDevice.d_destinationDenseInput),
            static_cast<std::size_t>(newDevice.destinationTotalElements) * sizeof(int),
            "cudaMalloc d_destinationDenseInput")) {
        return false;
    }

    if (!upload(newDevice.d_depDelay, dataset.depDelay.data(), delayBytes, "cudaMemcpy H2D d_depDelay")) {
        return false;
    }

    if (!upload(newDevice.d_arrDelay, dataset.arrDelay.data(), delayBytes, "cudaMemcpy H2D d_arrDelay")) {
        return false;
    }

    if (!upload(newDevice.d_tailNums, tailBuffer.data(), tailBytes, "cudaMemcpy H2D d_tailNums")) {
        return false;
    }

    if (newDevice.originTotalElements > 0 &&
        !upload(
            newDevice.d_originDenseInput,
            originDenseInput.data(),
            static_cast<std::size_t>(newDevice.originTotalElements) * sizeof(int),
            "cudaMemcpy H2D d_originDenseInput")) {
        return false;
    }

    if (newDevice.destinationTotalElements > 0 &&
        !upload(
            newDevice.d_destinationDenseInput,
            destinationDenseInput.data(),
            static_cast<std::size_t>(newDevice.destinationTotalElements) * sizeof(int),
            "cudaMemcpy H2D d_destinationDenseInput")) {
        return false;
    }

    releaseDeviceDataset(device);
    device = newDevice;
    errorMessage.clear();
    return true;
}

/*
    Resumenes de estado
*/

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
        const LaunchConfig launchConfig = computeLaunchConfig(static_cast<int>(getDatasetRowCount(appState.dataset)), appState.deviceProp);

        std::cout << "Sugerencia base: " << launchConfig.blocks
                  << " bloques x " << launchConfig.threadsPerBlock << " hilos\n";

        if (appState.device.rowCount > 0) {
            std::cout << "Dataset persistente en GPU listo para Fases 01, 02 y 04.\n";
        }
    }
}

/*
    Carga del dataset y preparacion del estado
*/

bool loadDatasetIntoState(AppState& appState, const std::string& datasetPath)
{
    /*
        Carga el CSV en host y, si hay GPU disponible, reconstruye el dataset
        persistente completo en device antes de sustituir el estado anterior.
    */
    DatasetColumns newDataset;
    LoadSummary newSummary;
    std::string errorMessage;

    if (!loadDataset(datasetPath, newDataset, newSummary, errorMessage)) {
        std::cout << "No se ha podido cargar el dataset.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    DeviceDataset newDevice;

    if (appState.deviceReady && !buildDeviceDataset(newDataset, newDevice, errorMessage)) {
        std::cout << "No se ha podido construir el dataset persistente en GPU.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    releaseDeviceDataset(appState.device);
    appState.datasetPath = datasetPath;
    appState.dataset = std::move(newDataset);
    appState.summary = newSummary;
    appState.device = newDevice;
    appState.datasetLoaded = true;

    printLoadSummary(appState);
    printGpuSummary(appState);
    return true;
}

bool promptAndLoadDataset(AppState& appState, bool allowCancel)
{
    /*
        Pide la ruta del CSV, resuelve la ruta por defecto si existe y repite
        el intento hasta cargar bien o cancelar, segun el contexto.
    */
    while (true) {
        std::string defaultPath;

        if (!appState.datasetPath.empty() && fileExists(appState.datasetPath)) {
            defaultPath = appState.datasetPath;
        } else if (fileExists("src/data/Airline_dataset.csv")) {
            defaultPath = "src/data/Airline_dataset.csv";
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
    /*
        Todas las fases GPU comparten la misma precondicion minima:

        - dataset cargado en host;
        - GPU CUDA accesible;
        - dataset persistente creado en device.
    */
    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado.\n";
        return false;
    }

    if (!appState.deviceReady) {
        std::cout << "No hay GPU CUDA disponible.\n";
        std::cout << "Motivo: " << appState.deviceErrorMessage << "\n";
        return false;
    }

    if (appState.device.rowCount <= 0) {
        std::cout << "El dataset persistente en GPU no esta disponible.\n";
        return false;
    }

    return true;
}

/*
    Fase 02: resumen CPU
*/

void printPhase2HostSummary(
    int threshold,
    int resultCount,
    const std::vector<int>& outDelayValues,
    const std::vector<char>& outTailNumBuffer)
{
    const char* label = threshold >= 0 ? "Retraso" : "Adelanto";

    std::cout << "\nResultados CPU:\n";
    std::cout << "Se han encontrado " << resultCount << " aviones\n";

    for (int i = 0; i < resultCount; ++i) {
        const char* tailNum = &outTailNumBuffer[static_cast<std::size_t>(i) * kPhase2TailNumStride];
        const int detectedValue = outDelayValues[static_cast<std::size_t>(i)];

        std::cout << "- Matricula " << tailNum
                  << " | " << label
                  << ": " << detectedValue << " minutos\n";
    }
}

/*
    Fases CUDA
*/

bool phase01(const AppState& appState, int threshold)
{
    /*
        Fase 01:

        - usa DEP_DELAY ya residente en GPU;
        - acepta un umbral firmado;
        - solo lanza el kernel y espera su salida por consola.
    */
    const int totalElements = appState.device.rowCount;
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    std::cout << "DEP_DELAY | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    phase1DepartureDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        appState.device.d_depDelay,
        totalElements,
        threshold);

    return finishKernel("phase1DepartureDelayKernel");
}

bool phase02(const AppState& appState, int threshold)
{
    /*
        Fase 02:

        - usa ARR_DELAY y TAIL_NUM ya residentes en GPU;
        - copia el umbral firmado a memoria constante;
        - recupera al host solo el subconjunto detectado por el kernel.
    */
    const int totalElements = appState.device.rowCount;
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    std::cout << "ARR_DELAY + TAIL_NUM | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    CUDA_RETURN_FALSE(cudaMemset(appState.device.d_phase2Count, 0, sizeof(int)));
    CUDA_RETURN_FALSE(copyPhase2ThresholdToConstant(threshold));

    phase2ArrivalDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        appState.device.d_arrDelay,
        appState.device.d_tailNums,
        totalElements,
        appState.device.d_phase2Count,
        appState.device.d_phase2OutDelayValues,
        appState.device.d_phase2OutTailNums);

    if (!finishKernel("phase2ArrivalDelayKernel")) {
        return false;
    }

    int resultCount = 0;
    CUDA_RETURN_FALSE(cudaMemcpy(&resultCount, appState.device.d_phase2Count, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> outDelayValues(static_cast<std::size_t>(resultCount));
    std::vector<char> outTailNumBuffer(static_cast<std::size_t>(resultCount) * kPhase2TailNumStride, '\0');

    if (resultCount > 0) {
        CUDA_RETURN_FALSE(cudaMemcpy(
            outDelayValues.data(),
            appState.device.d_phase2OutDelayValues,
            static_cast<std::size_t>(resultCount) * sizeof(int),
            cudaMemcpyDeviceToHost));

        CUDA_RETURN_FALSE(cudaMemcpy(
            outTailNumBuffer.data(),
            appState.device.d_phase2OutTailNums,
            static_cast<std::size_t>(resultCount) * kPhase2TailNumStride * sizeof(char),
            cudaMemcpyDeviceToHost));
    }

    printPhase2HostSummary(threshold, resultCount, outDelayValues, outTailNumBuffer);
    return true;
}

bool phase03AtomicVariant(
    Phase3AtomicVariant variant,
    int* deviceInput,
    int totalElements,
    bool isMax,
    const LaunchConfig& launchConfig,
    int& outResult)
{
    /*
        Ejecuta una de las tres variantes atomicas de la Fase 03 sobre el
        mismo vector de entrada ya compacto en GPU.
    */
    int* deviceResult = nullptr;
    const int initialValue = getReductionIdentity(isMax);
    std::size_t sharedBytes = 0;

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(int)), "cudaMalloc deviceResult")) {
        return false;
    }

    if (!cudaOk(cudaMemcpy(deviceResult, &initialValue, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy deviceResult")) {
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

    if (!finishKernel("Fase 03 atomica")) {
        cudaFree(deviceResult);
        return false;
    }

    if (!cudaOk(cudaMemcpy(&outResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy deviceResult")) {
        cudaFree(deviceResult);
        return false;
    }

    cudaFree(deviceResult);
    return true;
}

bool phase03ReductionVariant(int* deviceInput, int totalElements, bool isMax, const cudaDeviceProp& deviceProp, int& outResult)
{
    /*
        Variante 3.4:

        - reduce por bloques en GPU;
        - relanza sobre parciales;
        - cierra en CPU solo cuando el vector final ya tiene 10 elementos o menos.
    */
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

        if (!finishKernel("reductionPattern")) {
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
            "cudaMemcpy vector final Fase 03")) {
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

bool phase03(const AppState& appState, Phase3ColumnOption columnOption, ReductionTypeOption reductionOption)
{
    /*
        Orquestador completo de la Fase 03:

        - selecciona columna;
        - compacta ignorando NAN;
        - copia el vector entero a GPU;
        - ejecuta las cuatro variantes seguidas.
    */
    const std::vector<float>& sourceColumn = selectPhase3Column(appState.dataset, columnOption);

    std::vector<int> inputValues;
    inputValues.reserve(sourceColumn.size());

    for (std::size_t i = 0; i < sourceColumn.size(); ++i) {
        if (!std::isnan(sourceColumn[i])) {
            inputValues.push_back(static_cast<int>(sourceColumn[i]));
        }
    }

    const int totalElements = static_cast<int>(inputValues.size());

    if (totalElements <= 0) {
        std::cout << "No hay valores validos para la Fase 03.\n";
        return false;
    }

    const bool isMax = reductionOption == ReductionTypeOption::Maximum;
    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    const char* columnLabel = "DEP_DELAY";
    const char* reductionLabel = isMax ? "Maximo" : "Minimo";
    const char* reductionFunctionLabel = isMax ? "Max" : "Min";

    if (columnOption == Phase3ColumnOption::ArrivalDelay) {
        columnLabel = "ARR_DELAY";
    } else if (columnOption == Phase3ColumnOption::WeatherDelay) {
        columnLabel = "WEATHER_DELAY";
    }

    std::cout << columnLabel
              << " | " << reductionLabel
              << " | validos " << totalElements
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    int* deviceInput = nullptr;

    if (!cudaOk(
            cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputValues.size() * sizeof(int)),
            "cudaMalloc deviceInput Fase 03")) {
        return false;
    }

    if (!cudaOk(
            cudaMemcpy(deviceInput, inputValues.data(), inputValues.size() * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy deviceInput Fase 03")) {
        cudaFree(deviceInput);
        return false;
    }

    int simpleResult = 0;
    int basicResult = 0;
    int intermediateResult = 0;
    int reductionResult = 0;

    if (!phase03AtomicVariant(Phase3AtomicVariant::Simple, deviceInput, totalElements, isMax, launchConfig, simpleResult) ||
        !phase03AtomicVariant(Phase3AtomicVariant::Basic, deviceInput, totalElements, isMax, launchConfig, basicResult) ||
        !phase03AtomicVariant(
            Phase3AtomicVariant::Intermediate,
            deviceInput,
            totalElements,
            isMax,
            launchConfig,
            intermediateResult) ||
        !phase03ReductionVariant(deviceInput, totalElements, isMax, appState.deviceProp, reductionResult)) {
        cudaFree(deviceInput);
        return false;
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

    return true;
}

void printPhase4Histogram(
    HistogramAirportTypeOption airportType,
    int threshold,
    const std::vector<unsigned int>& histogram,
    const std::vector<int>& denseToSeqId,
    const std::unordered_map<int, std::string>& idToCode)
{
    const char* airportLabel = airportType == HistogramAirportTypeOption::Origin ? "origen" : "destino";
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

    std::cout << "\n(4) Histograma de aeropuertos de " << airportLabel << "\n";
    std::cout << "Num de aeropuertos encontrados: " << denseToSeqId.size() << "\n\n";

    for (std::size_t denseIndex = 0; denseIndex < histogram.size(); ++denseIndex) {
        if (histogram[denseIndex] < minimumCount) {
            continue;
        }

        const int seqId = denseToSeqId[denseIndex];
        std::unordered_map<int, std::string>::const_iterator codeIt = idToCode.find(seqId);
        const std::string airportCode = codeIt == idToCode.end() ? "" : codeIt->second;

        std::cout << airportCode << " (" << seqId << ") | " << histogram[denseIndex] << " ";

        int barLength = 0;
        const int maxBarWidth = 40;

        if (maximumShownCount > 0) {
            barLength = static_cast<int>(
                (static_cast<unsigned long long>(histogram[denseIndex]) * static_cast<unsigned long long>(maxBarWidth)) /
                static_cast<unsigned long long>(maximumShownCount));
        }

        if (barLength <= 0 && histogram[denseIndex] > 0) {
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

bool phase04(const AppState& appState, HistogramAirportTypeOption airportType, int threshold)
{
    /*
        Fase 04:

        - selecciona origen o destino dentro del dataset persistente;
        - lanza histograma compartido y fusion global;
        - devuelve el histograma a CPU para dibujarlo por consola.
    */
    const bool useOrigin = airportType == HistogramAirportTypeOption::Origin;
    const int totalElements = useOrigin ? appState.device.originTotalElements : appState.device.destinationTotalElements;
    const int totalBins = useOrigin ? appState.device.originTotalBins : appState.device.destinationTotalBins;
    const int* denseInput = useOrigin ? appState.device.d_originDenseInput : appState.device.d_destinationDenseInput;
    const std::vector<int>& denseToSeqId = useOrigin ? appState.device.originDenseToSeqId : appState.device.destinationDenseToSeqId;
    const std::unordered_map<int, std::string>& idToCode =
        useOrigin ? appState.dataset.originIdToCode : appState.dataset.destIdToCode;
    const char* airportLabel = useOrigin ? "origen" : "destino";

    if (totalElements <= 0 || totalBins <= 0 || denseInput == nullptr) {
        std::cout << "No hay datos validos para la Fase 04.\n";
        return false;
    }

    const std::size_t sharedBytes = static_cast<std::size_t>(totalBins) * sizeof(unsigned int);

    if (sharedBytes > static_cast<std::size_t>(appState.deviceProp.sharedMemPerBlock)) {
        std::cout << "El histograma no cabe en la memoria compartida por bloque de esta GPU.\n";
        return false;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);
    const LaunchConfig mergeLaunchConfig = computeLaunchConfig(totalBins, appState.deviceProp);

    std::cout << airportLabel
              << " | filas validas " << totalElements
              << " | bins " << totalBins
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    unsigned int* devicePartialHistograms = nullptr;
    unsigned int* deviceFinalHistogram = nullptr;

    const std::size_t partialBytes =
        static_cast<std::size_t>(launchConfig.blocks) * static_cast<std::size_t>(totalBins) * sizeof(unsigned int);
    const std::size_t finalBytes = static_cast<std::size_t>(totalBins) * sizeof(unsigned int);

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&devicePartialHistograms), partialBytes), "cudaMalloc devicePartialHistograms")) {
        return false;
    }

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&deviceFinalHistogram), finalBytes), "cudaMalloc deviceFinalHistogram")) {
        cudaFree(devicePartialHistograms);
        return false;
    }

    phase4SharedHistogramKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
        denseInput,
        totalElements,
        totalBins,
        devicePartialHistograms);

    if (!finishKernel("phase4SharedHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    phase4MergeHistogramKernel<<<mergeLaunchConfig.blocks, mergeLaunchConfig.threadsPerBlock>>>(
        devicePartialHistograms,
        launchConfig.blocks,
        totalBins,
        deviceFinalHistogram);

    if (!finishKernel("phase4MergeHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    std::vector<unsigned int> histogram(static_cast<std::size_t>(totalBins), 0U);

    if (!cudaOk(cudaMemcpy(histogram.data(), deviceFinalHistogram, finalBytes, cudaMemcpyDeviceToHost), "cudaMemcpy deviceFinalHistogram")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    cudaFree(devicePartialHistograms);
    cudaFree(deviceFinalHistogram);

    printPhase4Histogram(airportType, threshold, histogram, denseToSeqId, idToCode);
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

    pauseForEnter();

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
        case MainMenuOption::Phase1: {
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 01 - DEP_DELAY\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral firmado (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                break;
            }

            if (!phase01(appState, threshold)) {
                std::cout << "La Fase 01 no se ha podido completar.\n";
            }

            pauseForEnter();
            break;
        }

        case MainMenuOption::Phase2: {
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 02 - ARR_DELAY + TAIL_NUM\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral firmado (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                break;
            }

            if (!phase02(appState, threshold)) {
                std::cout << "La Fase 02 no se ha podido completar.\n";
            }

            pauseForEnter();
            break;
        }

        case MainMenuOption::Phase3: {
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 03 - Reduccion\n";
            std::cout << "1. DEP_DELAY  2. ARR_DELAY  3. WEATHER_DELAY\n";

            int columnValue = 0;
            int reductionValue = 0;

            if (!readIntegerInRange("Columna: ", "Debe introducir un numero entre 1 y 3, o X.", 1, 3, columnValue)) {
                break;
            }

            std::cout << "1. Maximo  2. Minimo\n";

            if (!readIntegerInRange("Reduccion: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, reductionValue)) {
                break;
            }

            if (!phase03(
                    appState,
                    static_cast<Phase3ColumnOption>(columnValue),
                    static_cast<ReductionTypeOption>(reductionValue))) {
                std::cout << "La Fase 03 no se ha podido completar.\n";
            }

            pauseForEnter();
            break;
        }

        case MainMenuOption::Phase4: {
            if (!canRunGpuPhase(appState)) {
                break;
            }

            std::cout << "\nFase 04 - Histograma de aeropuertos\n";
            std::cout << "1. Origen  2. Destino\n";

            int airportValue = 0;
            int threshold = 0;

            if (!readIntegerInRange("Tipo de aeropuerto: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, airportValue)) {
                break;
            }

            if (!readIntegerInRange("Umbral minimo (>= 0, X para volver): ", "Debe introducir un numero mayor o igual que 0, o X.", 0, INT_MAX, threshold)) {
                break;
            }

            if (!phase04(appState, static_cast<HistogramAirportTypeOption>(airportValue), threshold)) {
                std::cout << "La Fase 04 no se ha podido completar.\n";
            }

            pauseForEnter();
            break;
        }

        case MainMenuOption::ReloadCsv:
            promptAndLoadDataset(appState, true);
            break;

        case MainMenuOption::ShowStatus:
            if (appState.datasetLoaded) {
                printLoadSummary(appState);
            } else {
                std::cout << "\nDataset no cargado.\n";
            }

            printGpuSummary(appState);
            pauseForEnter();
            break;

        case MainMenuOption::Exit:
            keepRunning = false;
            break;
        }
    }

    releaseDeviceDataset(appState.device);
    std::cout << "\nAplicacion finalizada.\n";
    return 0;
}
