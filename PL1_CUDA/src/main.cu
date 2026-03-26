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

    Esta version quita gran parte del pegamento anterior y se acerca mas al
    estilo de cudasergi:

    - estado host/device guardado en globals simples;
    - dataset util subido una sola vez a GPU;
    - menos structs de orquestacion;
    - menos firmas largas pasando estado de un lado a otro.

    Se mantiene lo importante de la practica:

    - Fase 01 con DEP_DELAY;
    - Fase 02 con ARR_DELAY, TAIL_NUM, memoria constante y resumen CPU;
    - Fase 03 con DEP_DELAY, ARR_DELAY, WEATHER_DELAY y sus cuatro variantes;
    - Fase 03.4 iterando hasta dejar 10 valores o menos;
    - Fase 04 con bins densos por SEQ_ID y memoria compartida.
*/

#define CUDA_RETURN_FALSE(call) \
    do { \
        if (!cudaOk((call), #call)) return false; \
    } while (0)

struct LaunchConfig {
    int blocks = 0;
    int threadsPerBlock = 1;
};

enum class Phase3AtomicVariant {
    Simple,
    Basic,
    Intermediate
};

/*
    Estado global del programa.

    Se deja en globals a proposito para reducir tipado y evitar pasar structs
    grandes entre funciones.
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

namespace {

/*
    Helpers pequenos de texto y consola
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
    Helpers CUDA pequenos
*/

bool cudaOk(cudaError_t status, const char* context)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cout << "Error CUDA en " << context << ": "
              << cudaGetErrorString(status) << "\n";
    return false;
}

bool ejecutarKernelYEsperar(const char* context)
{
    if (!cudaOk(cudaGetLastError(), context)) {
        return false;
    }

    return cudaOk(cudaDeviceSynchronize(), context);
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

/*
    Construccion y liberacion del dataset GPU
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
            const int denseIndex = static_cast<int>(denseToSeqId.size());
            denseIndexBySeqId[seqId] = denseIndex;
            denseToSeqId.push_back(seqId);
            denseInput.push_back(denseIndex);
        } else {
            denseInput.push_back(found->second);
        }
    }
}

void liberarGPU()
{
    if (d_depDelay != nullptr) {
        cudaFree(d_depDelay);
        d_depDelay = nullptr;
    }

    if (d_arrDelay != nullptr) {
        cudaFree(d_arrDelay);
        d_arrDelay = nullptr;
    }

    if (d_tailNums != nullptr) {
        cudaFree(d_tailNums);
        d_tailNums = nullptr;
    }

    if (d_phase2Count != nullptr) {
        cudaFree(d_phase2Count);
        d_phase2Count = nullptr;
    }

    if (d_phase2OutDelayValues != nullptr) {
        cudaFree(d_phase2OutDelayValues);
        d_phase2OutDelayValues = nullptr;
    }

    if (d_phase2OutTailNums != nullptr) {
        cudaFree(d_phase2OutTailNums);
        d_phase2OutTailNums = nullptr;
    }

    if (d_originDenseInput != nullptr) {
        cudaFree(d_originDenseInput);
        d_originDenseInput = nullptr;
    }

    if (d_destinationDenseInput != nullptr) {
        cudaFree(d_destinationDenseInput);
        d_destinationDenseInput = nullptr;
    }

    g_rowCount = 0;
    g_originTotalElements = 0;
    g_originTotalBins = 0;
    g_destinationTotalElements = 0;
    g_destinationTotalBins = 0;
    g_originDenseToSeqId.clear();
    g_destinationDenseToSeqId.clear();
}

bool subirDatasetAGPU(const DatasetColumns& dataset, std::string& errorMessage)
{
    /*
        Sube a GPU solo lo que se reutiliza entre fases:

        - DEP_DELAY para Fase 01;
        - ARR_DELAY y TAIL_NUM para Fase 02;
        - entradas densas de origen y destino para Fase 04;
        - buffers persistentes de salida para Fase 02.
    */
    const int rowCount = static_cast<int>(dataset.depDelay.size());

    if (rowCount <= 0) {
        errorMessage = "No hay filas validas para construir el dataset en GPU.";
        return false;
    }

    std::vector<char> tailBuffer;
    std::vector<int> originDenseInput;
    std::vector<int> destinationDenseInput;
    std::vector<int> originDenseToSeqId;
    std::vector<int> destinationDenseToSeqId;

    buildTailBuffer(dataset.tailNum, tailBuffer);
    buildDenseInput(dataset.originSeqId, dataset.originIdToCode, originDenseToSeqId, originDenseInput);
    buildDenseInput(dataset.destSeqId, dataset.destIdToCode, destinationDenseToSeqId, destinationDenseInput);

    liberarGPU();

    const std::size_t delayBytes = static_cast<std::size_t>(rowCount) * sizeof(float);
    const std::size_t tailBytes = tailBuffer.size() * sizeof(char);
    const std::size_t outDelayBytes = static_cast<std::size_t>(rowCount) * sizeof(int);
    const std::size_t outTailBytes = static_cast<std::size_t>(rowCount) * kPhase2TailNumStride * sizeof(char);

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_depDelay), delayBytes), "cudaMalloc d_depDelay") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_arrDelay), delayBytes), "cudaMalloc d_arrDelay") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_tailNums), tailBytes), "cudaMalloc d_tailNums") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_phase2Count), sizeof(int)), "cudaMalloc d_phase2Count") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_phase2OutDelayValues), outDelayBytes), "cudaMalloc d_phase2OutDelayValues") ||
        !cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_phase2OutTailNums), outTailBytes), "cudaMalloc d_phase2OutTailNums")) {
        liberarGPU();
        errorMessage = "No se ha podido reservar la memoria principal de GPU.";
        return false;
    }

    if (!originDenseInput.empty() &&
        !cudaOk(
            cudaMalloc(
                reinterpret_cast<void**>(&d_originDenseInput),
                static_cast<std::size_t>(originDenseInput.size()) * sizeof(int)),
            "cudaMalloc d_originDenseInput")) {
        liberarGPU();
        errorMessage = "No se ha podido reservar d_originDenseInput.";
        return false;
    }

    if (!destinationDenseInput.empty() &&
        !cudaOk(
            cudaMalloc(
                reinterpret_cast<void**>(&d_destinationDenseInput),
                static_cast<std::size_t>(destinationDenseInput.size()) * sizeof(int)),
            "cudaMalloc d_destinationDenseInput")) {
        liberarGPU();
        errorMessage = "No se ha podido reservar d_destinationDenseInput.";
        return false;
    }

    if (!cudaOk(cudaMemcpy(d_depDelay, dataset.depDelay.data(), delayBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_depDelay") ||
        !cudaOk(cudaMemcpy(d_arrDelay, dataset.arrDelay.data(), delayBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_arrDelay") ||
        !cudaOk(cudaMemcpy(d_tailNums, tailBuffer.data(), tailBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_tailNums")) {
        liberarGPU();
        errorMessage = "No se han podido copiar las columnas base a GPU.";
        return false;
    }

    if (!originDenseInput.empty() &&
        !cudaOk(
            cudaMemcpy(
                d_originDenseInput,
                originDenseInput.data(),
                static_cast<std::size_t>(originDenseInput.size()) * sizeof(int),
                cudaMemcpyHostToDevice),
            "cudaMemcpy H2D d_originDenseInput")) {
        liberarGPU();
        errorMessage = "No se ha podido copiar d_originDenseInput.";
        return false;
    }

    if (!destinationDenseInput.empty() &&
        !cudaOk(
            cudaMemcpy(
                d_destinationDenseInput,
                destinationDenseInput.data(),
                static_cast<std::size_t>(destinationDenseInput.size()) * sizeof(int),
                cudaMemcpyHostToDevice),
            "cudaMemcpy H2D d_destinationDenseInput")) {
        liberarGPU();
        errorMessage = "No se ha podido copiar d_destinationDenseInput.";
        return false;
    }

    g_rowCount = rowCount;
    g_originTotalElements = static_cast<int>(originDenseInput.size());
    g_originTotalBins = static_cast<int>(originDenseToSeqId.size());
    g_destinationTotalElements = static_cast<int>(destinationDenseInput.size());
    g_destinationTotalBins = static_cast<int>(destinationDenseToSeqId.size());
    g_originDenseToSeqId = std::move(originDenseToSeqId);
    g_destinationDenseToSeqId = std::move(destinationDenseToSeqId);

    errorMessage.clear();
    return true;
}

/*
    Resumenes y estado
*/

void printLoadSummary()
{
    std::cout << "\n=== Fase 0 ===\n";
    std::cout << "Ruta: " << g_datasetPath << "\n";
    std::cout << "Filas de datos leidas: " << g_summary.rowsRead << "\n";
    std::cout << "Filas almacenadas: " << g_summary.storedRows << "\n";
    std::cout << "Filas descartadas: " << g_summary.discardedRows << "\n";

    std::cout << "\nValores ausentes detectados:\n";
    std::cout << "- TAIL_NUM: " << g_summary.missingTailNum << "\n";
    std::cout << "- ORIGIN_SEQ_ID: " << g_summary.missingOriginSeqId << "\n";
    std::cout << "- ORIGIN_AIRPORT: " << g_summary.missingOriginAirportCode << "\n";
    std::cout << "- DEST_SEQ_ID: " << g_summary.missingDestSeqId << "\n";
    std::cout << "- DEST_AIRPORT: " << g_summary.missingDestAirportCode << "\n";
    std::cout << "- DEP_DELAY: " << g_summary.missingDepDelay << "\n";
    std::cout << "- ARR_DELAY: " << g_summary.missingArrDelay << "\n";
    std::cout << "- WEATHER_DELAY: " << g_summary.missingWeatherDelay << "\n";

    std::cout << "\nCategorias detectadas:\n";
    std::cout << "- Aeropuertos unicos de origen por SEQ_ID: " << g_summary.uniqueOriginSeqIds << "\n";
    std::cout << "- Aeropuertos unicos de destino por SEQ_ID: " << g_summary.uniqueDestinationSeqIds << "\n";
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

/*
    Carga del dataset
*/

bool cargarDataset(const std::string& datasetPath)
{
    DatasetColumns newDataset;
    LoadSummary newSummary;
    std::string errorMessage;

    if (!loadDataset(datasetPath, newDataset, newSummary, errorMessage)) {
        std::cout << "No se ha podido cargar el dataset.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    if (g_deviceReady && !subirDatasetAGPU(newDataset, errorMessage)) {
        std::cout << "No se ha podido cargar el dataset en GPU.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    g_datasetPath = datasetPath;
    g_dataset = std::move(newDataset);
    g_summary = newSummary;
    g_datasetLoaded = true;

    printLoadSummary();
    printGpuSummary();
    return true;
}

bool promptAndLoadDataset(bool allowCancel)
{
    while (true) {
        std::string defaultPath;

        if (!g_datasetPath.empty() && fileExists(g_datasetPath)) {
            defaultPath = g_datasetPath;
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

        if (cargarDataset(selectedPath)) {
            return true;
        }
    }
}

bool datasetListoParaGPU()
{
    if (!g_datasetLoaded) {
        std::cout << "No hay dataset cargado.\n";
        return false;
    }

    if (!g_deviceReady) {
        std::cout << "No hay GPU CUDA disponible.\n";
        std::cout << "Motivo: " << g_deviceErrorMessage << "\n";
        return false;
    }

    if (g_rowCount <= 0) {
        std::cout << "El dataset en GPU no esta disponible.\n";
        return false;
    }

    return true;
}

/*
    Fase 02: resumen CPU
*/

void printPhase2HostSummary(int threshold, int resultCount, const std::vector<int>& outDelayValues, const std::vector<char>& outTailNumBuffer)
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
    Fases 01 y 02
*/

bool phase01(int threshold)
{
    const LaunchConfig launchConfig = computeLaunchConfig(g_rowCount);

    std::cout << "DEP_DELAY | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    phase1DepartureDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        d_depDelay,
        g_rowCount,
        threshold);

    return ejecutarKernelYEsperar("phase1DepartureDelayKernel");
}

bool phase02(int threshold)
{
    const LaunchConfig launchConfig = computeLaunchConfig(g_rowCount);

    std::cout << "ARR_DELAY + TAIL_NUM | umbral " << threshold
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    CUDA_RETURN_FALSE(cudaMemset(d_phase2Count, 0, sizeof(int)));
    CUDA_RETURN_FALSE(copyPhase2ThresholdToConstant(threshold));

    phase2ArrivalDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        d_arrDelay,
        d_tailNums,
        g_rowCount,
        d_phase2Count,
        d_phase2OutDelayValues,
        d_phase2OutTailNums);

    if (!ejecutarKernelYEsperar("phase2ArrivalDelayKernel")) {
        return false;
    }

    int resultCount = 0;
    CUDA_RETURN_FALSE(cudaMemcpy(&resultCount, d_phase2Count, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> outDelayValues(static_cast<std::size_t>(resultCount));
    std::vector<char> outTailNumBuffer(static_cast<std::size_t>(resultCount) * kPhase2TailNumStride, '\0');

    if (resultCount > 0) {
        CUDA_RETURN_FALSE(cudaMemcpy(
            outDelayValues.data(),
            d_phase2OutDelayValues,
            static_cast<std::size_t>(resultCount) * sizeof(int),
            cudaMemcpyDeviceToHost));

        CUDA_RETURN_FALSE(cudaMemcpy(
            outTailNumBuffer.data(),
            d_phase2OutTailNums,
            static_cast<std::size_t>(resultCount) * kPhase2TailNumStride * sizeof(char),
            cudaMemcpyDeviceToHost));
    }

    printPhase2HostSummary(threshold, resultCount, outDelayValues, outTailNumBuffer);
    return true;
}

/*
    Helpers de reduccion para Fase 03
*/

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

    if (!ejecutarKernelYEsperar("Fase 03 atomica")) {
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

        if (!ejecutarKernelYEsperar("reductionPattern")) {
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

bool phase03(int columnOption, int reductionOption)
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
        return false;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(totalElements);

    std::cout << columnLabel
              << " | " << reductionLabel
              << " | validos " << totalElements
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    int* deviceInput = nullptr;

    if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputValues.size() * sizeof(int)), "cudaMalloc deviceInput Fase 03")) {
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
        !phase03AtomicVariant(Phase3AtomicVariant::Intermediate, deviceInput, totalElements, isMax, launchConfig, intermediateResult) ||
        !phase03ReductionVariant(deviceInput, totalElements, isMax, reductionResult)) {
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

/*
    Fase 04
*/

void printPhase4Histogram(
    const char* airportLabel,
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

bool phase04(int airportOption, int threshold)
{
    const bool useOrigin = airportOption == 1;
    const int totalElements = useOrigin ? g_originTotalElements : g_destinationTotalElements;
    const int totalBins = useOrigin ? g_originTotalBins : g_destinationTotalBins;
    const int* denseInput = useOrigin ? d_originDenseInput : d_destinationDenseInput;
    const std::vector<int>& denseToSeqId = useOrigin ? g_originDenseToSeqId : g_destinationDenseToSeqId;
    const std::unordered_map<int, std::string>& idToCode = useOrigin ? g_dataset.originIdToCode : g_dataset.destIdToCode;
    const char* airportLabel = useOrigin ? "origen" : "destino";

    if (totalElements <= 0 || totalBins <= 0 || denseInput == nullptr) {
        std::cout << "No hay datos validos para la Fase 04.\n";
        return false;
    }

    const std::size_t sharedBytes = static_cast<std::size_t>(totalBins) * sizeof(unsigned int);

    if (sharedBytes > static_cast<std::size_t>(g_deviceProp.sharedMemPerBlock)) {
        std::cout << "El histograma no cabe en la memoria compartida por bloque de esta GPU.\n";
        return false;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(totalElements);
    const LaunchConfig mergeLaunchConfig = computeLaunchConfig(totalBins);

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

    if (!ejecutarKernelYEsperar("phase4SharedHistogramKernel")) {
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        return false;
    }

    phase4MergeHistogramKernel<<<mergeLaunchConfig.blocks, mergeLaunchConfig.threadsPerBlock>>>(
        devicePartialHistograms,
        launchConfig.blocks,
        totalBins,
        deviceFinalHistogram);

    if (!ejecutarKernelYEsperar("phase4MergeHistogramKernel")) {
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

    printPhase4Histogram(airportLabel, threshold, histogram, denseToSeqId, idToCode);
    return true;
}

} // namespace

int main()
{
    std::cout << "========================================\n";
    std::cout << " PL1 CUDA - US Airline Dataset Toolkit\n";
    std::cout << "========================================\n";

    queryGpuInfo();
    printGpuSummary();

    if (!promptAndLoadDataset(true)) {
        liberarGPU();
        std::cout << "Saliendo sin cargar dataset.\n";
        return 0;
    }

    pauseForEnter();

    bool keepRunning = true;

    while (keepRunning) {
        std::string optionInput;

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

            std::getline(std::cin, optionInput);
            optionInput = trimWhitespace(optionInput);

            if (optionInput == "1" ||
                optionInput == "2" ||
                optionInput == "3" ||
                optionInput == "4" ||
                optionInput == "R" || optionInput == "r" ||
                optionInput == "I" || optionInput == "i" ||
                isCancelToken(optionInput)) {
                break;
            }

            std::cout << "Opcion no valida.\n";
        }

        if (optionInput == "1") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 01 - DEP_DELAY\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral firmado (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                continue;
            }

            if (!phase01(threshold)) {
                std::cout << "La Fase 01 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "2") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 02 - ARR_DELAY + TAIL_NUM\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral firmado (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                continue;
            }

            if (!phase02(threshold)) {
                std::cout << "La Fase 02 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "3") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 03 - Reduccion\n";
            std::cout << "1. DEP_DELAY  2. ARR_DELAY  3. WEATHER_DELAY\n";

            int columnValue = 0;
            int reductionValue = 0;

            if (!readIntegerInRange("Columna: ", "Debe introducir un numero entre 1 y 3, o X.", 1, 3, columnValue)) {
                continue;
            }

            std::cout << "1. Maximo  2. Minimo\n";

            if (!readIntegerInRange("Reduccion: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, reductionValue)) {
                continue;
            }

            if (!phase03(columnValue, reductionValue)) {
                std::cout << "La Fase 03 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "4") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 04 - Histograma de aeropuertos\n";
            std::cout << "1. Origen  2. Destino\n";

            int airportValue = 0;
            int threshold = 0;

            if (!readIntegerInRange("Tipo de aeropuerto: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, airportValue)) {
                continue;
            }

            if (!readIntegerInRange("Umbral minimo (>= 0, X para volver): ", "Debe introducir un numero mayor o igual que 0, o X.", 0, INT_MAX, threshold)) {
                continue;
            }

            if (!phase04(airportValue, threshold)) {
                std::cout << "La Fase 04 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "R" || optionInput == "r") {
            promptAndLoadDataset(true);
            continue;
        }

        if (optionInput == "I" || optionInput == "i") {
            if (g_datasetLoaded) {
                printLoadSummary();
            } else {
                std::cout << "\nDataset no cargado.\n";
            }

            printGpuSummary();
            pauseForEnter();
            continue;
        }

        keepRunning = false;
    }

    liberarGPU();
    std::cout << "\nAplicacion finalizada.\n";
    return 0;
}
