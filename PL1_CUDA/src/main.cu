#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "cli_utils.h"
#include "csv_reader.h"
#include "kernels.cuh"

/*
    main.cu

    Este archivo coordina el estado actual del programa. Hoy la aplicacion
    implementa de forma real:

    - la Fase 0 de lectura y limpieza;
    - la Fase 01 de retraso en despegues;
    - la Fase 02 de retraso en aterrizajes.

    Flujo general actual:

    1. mostrar banner;
    2. detectar la GPU CUDA disponible;
    3. pedir la ruta del CSV y cargarlo;
    4. mostrar resumen de la limpieza y del hardware;
    5. entrar en un menu principal persistente;
    6. permitir ejecutar las fases 01 y 02;
    7. dejar preparadas las fases 03 y 04;
    8. permitir recargar el dataset y consultar el estado.

    La CPU se encarga de:

    - cargar y limpiar el CSV;
    - preparar buffers sencillos para GPU;
    - lanzar kernels;
    - recuperar resultados cuando la fase lo exige.

    La GPU se usa ya en las fases 01 y 02 para realizar el filtrado pedido por
    el enunciado.
*/

/*
    LaunchConfig

    Estructura minima para describir una configuracion de lanzamiento sugerida.
    Aunque los kernels definitivos aun no esten conectados al flujo principal,
    el programa ya calcula y muestra estos valores para no depender de un
    numero fijo "porque si" y para respetar la idea del enunciado.
*/
struct LaunchConfig {
    int blocks;
    int threadsPerBlock;
};

/*
    AppState

    Estado global de la aplicacion durante la ejecucion interactiva.

    Contiene:

    - la ruta del dataset que el usuario esta usando;
    - el resultado de la carga y limpieza actual;
    - el estado de disponibilidad de CUDA;
    - la informacion del dispositivo detectado.

    Esta estructura evita pasar muchos parametros sueltos entre funciones del
    menu y deja claro que todas esas funciones trabajan sobre el mismo contexto.
*/
struct AppState {
    std::string datasetPath;
    bool datasetLoaded = false;
    CsvLoadResult loadResult;

    bool deviceReady = false;
    cudaDeviceProp deviceProp{};
    std::string deviceErrorMessage;
};

namespace {

/*
    appendCandidateIfMissing

    Inserta una ruta candidata en un vector solo si:

    - no es vacia;
    - todavia no existe en la lista.

    Se usa para construir la lista de rutas sugeridas del dataset sin
    duplicados innecesarios en la interfaz.
*/
void appendCandidateIfMissing(std::vector<std::string>& candidates, const std::string& candidate)
{
    if (candidate.empty()) {
        return;
    }

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i] == candidate) {
            return;
        }
    }

    candidates.push_back(candidate);
}

/*
    buildDatasetCandidates

    Construye la lista de rutas que la CLI mostrara como posibles ubicaciones
    del CSV. Se prioriza:

    - la ultima ruta activa del usuario;
    - la ruta de ejemplo dentro de src/data.

    Se elimina la antigua ruta data/Airline_dataset.csv para que el proyecto
    tenga una unica referencia por defecto y no arrastre configuraciones
    antiguas o binarios viejos.
*/
std::vector<std::string> buildDatasetCandidates(const AppState& appState)
{
    std::vector<std::string> candidates;

    appendCandidateIfMissing(candidates, appState.datasetPath);
    appendCandidateIfMissing(candidates, "src/data/Airline_dataset.csv");

    return candidates;
}

/*
    queryGpuInfo

    Comprueba si existe al menos un dispositivo CUDA accesible y, si existe,
    rellena la estructura cudaDeviceProp del dispositivo 0.

    No lanza excepciones ni termina el programa por si sola. En su lugar,
    devuelve true/false y un mensaje de error explicativo para que la capa de
    interfaz pueda informar al usuario.
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

/*
    computeLaunchConfig

    Calcula una configuracion sencilla de lanzamiento para un vector de tamano
    totalElements usando el hardware detectado. La idea es dejar preparada una
    base razonable para las fases futuras:

    - usar hasta 256 hilos por bloque si el hardware lo permite;
    - reducir ese valor si la GPU soporta menos;
    - calcular el numero minimo de bloques que cubre el vector completo.
*/
LaunchConfig computeLaunchConfig(int totalElements, const cudaDeviceProp& deviceProp)
{
    LaunchConfig launchConfig{};

    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock > 0 ? deviceProp.maxThreadsPerBlock : 1;
    launchConfig.threadsPerBlock = maxThreadsPerBlock < 256 ? maxThreadsPerBlock : 256;

    // Si todavia no hay datos, mostramos 0 bloques para evitar sugerencias
    // enganosas. Si ya hay dataset, calculamos el techo entero habitual.
    if (totalElements > 0) {
        launchConfig.blocks = (totalElements + launchConfig.threadsPerBlock - 1) / launchConfig.threadsPerBlock;
    } else {
        launchConfig.blocks = 0;
    }

    return launchConfig;
}

/*
    reportCudaFailure

    Helper minimo para centralizar la impresion de errores CUDA. La funcion
    devuelve true si la llamada ha ido bien y false si ha fallado, de forma que
    el codigo de las fases pueda encadenar comprobaciones sin repetir siempre
    el mismo bloque de salida por consola.
*/
bool reportCudaFailure(cudaError_t status, const char* context)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cout << "\nError CUDA en " << context << ": "
              << cudaGetErrorString(status) << "\n";
    return false;
}

/*
    getDelayFilterModeLabel

    Convierte el enum usado en la CLI y en el host a una etiqueta legible para
    mostrar resumentes de configuracion y mantener el flujo facil de seguir.
*/
const char* getDelayFilterModeLabel(DelayFilterMode mode)
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

/*
    detectDelayFilterLabel

    Etiqueta un valor concreto segun el umbral absoluto y el modo elegido.
    Esto permite que los resumentes host y device describan correctamente cada
    fila detectada, especialmente cuando el modo es "ambos".
*/
const char* detectDelayFilterLabel(int value, DelayFilterMode mode, int threshold)
{
    if (mode == DelayFilterMode::Delay) {
        return "Retraso";
    }

    if (mode == DelayFilterMode::Advance) {
        return "Adelanto";
    }

    // En modo "ambos" el dato concreto decide la etiqueta final.
    if (value >= threshold) {
        return "Retraso";
    }

    return "Adelanto";
}

/*
    releaseDeviceAllocation

    Libera un puntero device solo si realmente se habia reservado. Se usa para
    simplificar la limpieza de errores sin duplicar comprobaciones nulas en
    cada fase.
*/
void releaseDeviceAllocation(void* devicePointer)
{
    if (devicePointer != nullptr) {
        cudaFree(devicePointer);
    }
}

/*
    truncateTowardZero

    La practica pide trabajar con enteros cuando estas fases lo necesiten.
    El cast a int en C/C++ trunca hacia cero, que es exactamente la semantica
    sencilla que queremos mantener en host.
*/
int truncateTowardZero(float value)
{
    return static_cast<int>(value);
}

/*
    buildIntDelayBuffer

    Convierte una columna float del dataset en dos buffers paralelos:

    - outValues: retrasos truncados a entero;
    - outValidMask: mascara 0/1 para indicar si el dato original era valido.

    Se evita compactar el dataset para conservar el mismo indice global que
    usan las filas del CSV. Asi, el hilo CUDA i sigue representando la fila i.
*/
void buildIntDelayBuffer(
    const std::vector<float>& source,
    std::vector<int>& outValues,
    std::vector<unsigned char>& outValidMask)
{
    outValues.resize(source.size());
    outValidMask.resize(source.size());

    for (std::size_t i = 0; i < source.size(); ++i) {
        // Si el valor original es NAN, dejamos una mascara a 0 y un valor
        // neutro que nunca deberia usarse porque el kernel lo ignorara.
        if (std::isnan(source[i])) {
            outValues[i] = 0;
            outValidMask[i] = 0;
            continue;
        }

        outValues[i] = truncateTowardZero(source[i]);
        outValidMask[i] = 1;
    }
}

/*
    buildTailNumFixedBuffer

    Linealiza el vector de matriculas en un unico buffer de chars con stride
    fijo. Esta es la forma mas simple de llevar strings a GPU sin introducir
    estructuras mas complejas de lo necesario.
*/
void buildTailNumFixedBuffer(const std::vector<std::string>& source, std::vector<char>& outBuffer)
{
    outBuffer.assign(source.size() * kPhase2TailNumStride, '\0');

    for (std::size_t row = 0; row < source.size(); ++row) {
        const std::string& tailNum = source[row];
        char* outputCell = &outBuffer[row * kPhase2TailNumStride];

        // Copiamos como maximo stride - 1 para reservar siempre un '\0' final.
        const std::size_t maxCharacters = static_cast<std::size_t>(kPhase2TailNumStride - 1);
        const std::size_t charactersToCopy =
            tailNum.size() < maxCharacters ? tailNum.size() : maxCharacters;

        for (std::size_t i = 0; i < charactersToCopy; ++i) {
            outputCell[i] = tailNum[i];
        }

        outputCell[charactersToCopy] = '\0';
    }
}

/*
    printPhase2HostSummary

    La Fase 02 debe devolver al host el numero de aviones detectados y arrays
    simples con matriculas y tiempos. Esta funcion muestra ese resumen final en
    CPU utilizando justo esos buffers ya copiados desde device.
*/
void printPhase2HostSummary(
    DelayFilterMode mode,
    int threshold,
    int resultCount,
    const std::vector<int>& outDelayValues,
    const std::vector<char>& outTailNumBuffer)
{
    std::cout << "\nResultados completados de calcular en la CPU:\n";
    std::cout << "Se han encontrado " << resultCount << " aviones\n";

    for (int i = 0; i < resultCount; ++i) {
        const char* tailNum = &outTailNumBuffer[static_cast<std::size_t>(i) * kPhase2TailNumStride];
        const int detectedValue = outDelayValues[static_cast<std::size_t>(i)];
        const char* detectedLabel = detectDelayFilterLabel(detectedValue, mode, threshold);

        std::cout << "- Matricula " << tailNum
                  << "  " << detectedLabel << ": "
                  << detectedValue
                  << " minutos\n";
    }
}

/*
    runPhase1Computation

    Implementacion host de la Fase 01. Su responsabilidad es preparar buffers
    sencillos, reservar memoria CUDA, lanzar el kernel y sincronizar para que
    la salida por consola desde GPU se vea antes de volver al menu.
*/
bool runPhase1Computation(const AppState& appState, DelayFilterMode mode, int threshold)
{
    const DatasetColumns& dataset = appState.loadResult.dataset;

    std::vector<int> depDelayValues;
    std::vector<unsigned char> depDelayValidMask;
    buildIntDelayBuffer(dataset.depDelay, depDelayValues, depDelayValidMask);

    const int totalElements = static_cast<int>(depDelayValues.size());

    if (totalElements <= 0) {
        std::cout << "No hay datos disponibles para ejecutar la Fase 01.\n";
        return false;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    int* deviceDelayValues = nullptr;
    unsigned char* deviceValidMask = nullptr;

    const std::size_t delayBufferBytes = depDelayValues.size() * sizeof(int);
    const std::size_t validMaskBytes = depDelayValidMask.size() * sizeof(unsigned char);

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceDelayValues), delayBufferBytes),
        "cudaMalloc deviceDelayValues (Fase 01)")) {
        return false;
    }

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceValidMask), validMaskBytes),
        "cudaMalloc deviceValidMask (Fase 01)")) {
        releaseDeviceAllocation(deviceDelayValues);
        return false;
    }

    if (!reportCudaFailure(cudaMemcpy(
        deviceDelayValues,
        depDelayValues.data(),
        delayBufferBytes,
        cudaMemcpyHostToDevice),
        "cudaMemcpy H2D deviceDelayValues (Fase 01)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        return false;
    }

    if (!reportCudaFailure(cudaMemcpy(
        deviceValidMask,
        depDelayValidMask.data(),
        validMaskBytes,
        cudaMemcpyHostToDevice),
        "cudaMemcpy H2D deviceValidMask (Fase 01)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        return false;
    }

    std::cout << "\nEjecutando Fase 01 en GPU...\n";

    phase1DepartureDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        deviceDelayValues,
        deviceValidMask,
        totalElements,
        static_cast<int>(mode),
        threshold);

    // Primero comprobamos errores de lanzamiento y despues sincronizamos para
    // vaciar los printf de GPU antes de volver al menu.
    if (!reportCudaFailure(cudaGetLastError(), "lanzamiento phase1DepartureDelayKernel")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        return false;
    }

    if (!reportCudaFailure(cudaDeviceSynchronize(), "cudaDeviceSynchronize Fase 01")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        return false;
    }

    releaseDeviceAllocation(deviceDelayValues);
    releaseDeviceAllocation(deviceValidMask);

    return true;
}

/*
    runPhase2Computation

    Implementacion host de la Fase 02. Ademas del filtrado en GPU, esta fase
    debe recuperar al host:

    - el numero total de resultados;
    - un array simple de retrasos;
    - un array simple de matriculas.
*/
bool runPhase2Computation(const AppState& appState, DelayFilterMode mode, int threshold)
{
    const DatasetColumns& dataset = appState.loadResult.dataset;

    std::vector<int> arrDelayValues;
    std::vector<unsigned char> arrDelayValidMask;
    std::vector<char> tailNumBuffer;

    buildIntDelayBuffer(dataset.arrDelay, arrDelayValues, arrDelayValidMask);
    buildTailNumFixedBuffer(dataset.tailNum, tailNumBuffer);

    const int totalElements = static_cast<int>(arrDelayValues.size());

    if (totalElements <= 0) {
        std::cout << "No hay datos disponibles para ejecutar la Fase 02.\n";
        return false;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(totalElements, appState.deviceProp);

    int* deviceDelayValues = nullptr;
    unsigned char* deviceValidMask = nullptr;
    char* deviceTailNumBuffer = nullptr;
    int* deviceOutCount = nullptr;
    int* deviceOutDelayValues = nullptr;
    char* deviceOutTailNumBuffer = nullptr;

    const std::size_t delayBufferBytes = arrDelayValues.size() * sizeof(int);
    const std::size_t validMaskBytes = arrDelayValidMask.size() * sizeof(unsigned char);
    const std::size_t tailNumBufferBytes = tailNumBuffer.size() * sizeof(char);

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceDelayValues), delayBufferBytes),
        "cudaMalloc deviceDelayValues (Fase 02)")) {
        return false;
    }

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceValidMask), validMaskBytes),
        "cudaMalloc deviceValidMask (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        return false;
    }

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceTailNumBuffer), tailNumBufferBytes),
        "cudaMalloc deviceTailNumBuffer (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        return false;
    }

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceOutCount), sizeof(int)),
        "cudaMalloc deviceOutCount (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        return false;
    }

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceOutDelayValues), delayBufferBytes),
        "cudaMalloc deviceOutDelayValues (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        return false;
    }

    if (!reportCudaFailure(cudaMalloc(reinterpret_cast<void**>(&deviceOutTailNumBuffer), tailNumBufferBytes),
        "cudaMalloc deviceOutTailNumBuffer (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        return false;
    }

    if (!reportCudaFailure(cudaMemcpy(
        deviceDelayValues,
        arrDelayValues.data(),
        delayBufferBytes,
        cudaMemcpyHostToDevice),
        "cudaMemcpy H2D deviceDelayValues (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    if (!reportCudaFailure(cudaMemcpy(
        deviceValidMask,
        arrDelayValidMask.data(),
        validMaskBytes,
        cudaMemcpyHostToDevice),
        "cudaMemcpy H2D deviceValidMask (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    if (!reportCudaFailure(cudaMemcpy(
        deviceTailNumBuffer,
        tailNumBuffer.data(),
        tailNumBufferBytes,
        cudaMemcpyHostToDevice),
        "cudaMemcpy H2D deviceTailNumBuffer (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    if (!reportCudaFailure(cudaMemset(deviceOutCount, 0, sizeof(int)),
        "cudaMemset deviceOutCount (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    if (!reportCudaFailure(copyPhase2FilterConfigToConstant(static_cast<int>(mode), threshold),
        "copyPhase2FilterConfigToConstant (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    std::cout << "\nEjecutando Fase 02 en GPU...\n";

    phase2ArrivalDelayKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock>>>(
        deviceDelayValues,
        deviceValidMask,
        deviceTailNumBuffer,
        totalElements,
        deviceOutCount,
        deviceOutDelayValues,
        deviceOutTailNumBuffer);

    if (!reportCudaFailure(cudaGetLastError(), "lanzamiento phase2ArrivalDelayKernel")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    if (!reportCudaFailure(cudaDeviceSynchronize(), "cudaDeviceSynchronize Fase 02")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    int resultCount = 0;

    if (!reportCudaFailure(cudaMemcpy(
        &resultCount,
        deviceOutCount,
        sizeof(int),
        cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H deviceOutCount (Fase 02)")) {
        releaseDeviceAllocation(deviceDelayValues);
        releaseDeviceAllocation(deviceValidMask);
        releaseDeviceAllocation(deviceTailNumBuffer);
        releaseDeviceAllocation(deviceOutCount);
        releaseDeviceAllocation(deviceOutDelayValues);
        releaseDeviceAllocation(deviceOutTailNumBuffer);
        return false;
    }

    std::vector<int> outDelayValues(static_cast<std::size_t>(resultCount));
    std::vector<char> outTailNumBuffer(static_cast<std::size_t>(resultCount) * kPhase2TailNumStride, '\0');

    // Copiamos solo la parte realmente usada de los arrays de salida.
    if (resultCount > 0) {
        if (!reportCudaFailure(cudaMemcpy(
            outDelayValues.data(),
            deviceOutDelayValues,
            static_cast<std::size_t>(resultCount) * sizeof(int),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H deviceOutDelayValues (Fase 02)")) {
            releaseDeviceAllocation(deviceDelayValues);
            releaseDeviceAllocation(deviceValidMask);
            releaseDeviceAllocation(deviceTailNumBuffer);
            releaseDeviceAllocation(deviceOutCount);
            releaseDeviceAllocation(deviceOutDelayValues);
            releaseDeviceAllocation(deviceOutTailNumBuffer);
            return false;
        }

        if (!reportCudaFailure(cudaMemcpy(
            outTailNumBuffer.data(),
            deviceOutTailNumBuffer,
            static_cast<std::size_t>(resultCount) * kPhase2TailNumStride * sizeof(char),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H deviceOutTailNumBuffer (Fase 02)")) {
            releaseDeviceAllocation(deviceDelayValues);
            releaseDeviceAllocation(deviceValidMask);
            releaseDeviceAllocation(deviceTailNumBuffer);
            releaseDeviceAllocation(deviceOutCount);
            releaseDeviceAllocation(deviceOutDelayValues);
            releaseDeviceAllocation(deviceOutTailNumBuffer);
            return false;
        }
    }

    releaseDeviceAllocation(deviceDelayValues);
    releaseDeviceAllocation(deviceValidMask);
    releaseDeviceAllocation(deviceTailNumBuffer);
    releaseDeviceAllocation(deviceOutCount);
    releaseDeviceAllocation(deviceOutDelayValues);
    releaseDeviceAllocation(deviceOutTailNumBuffer);

    printPhase2HostSummary(mode, threshold, resultCount, outDelayValues, outTailNumBuffer);
    return true;
}

/*
    printLoadSummary

    Muestra por consola el resultado completo de la Fase 0. Esta salida es
    importante porque hace visible que ha ocurrido con el CSV:

    - cuantas filas se han leido;
    - cuantas se han almacenado;
    - cuantas se han descartado;
    - cuantos faltantes hay por columna;
    - cuantas categorias unicas se han encontrado.
*/
void printLoadSummary(const CsvLoadResult& loadResult)
{
    const CsvLoadStats& stats = loadResult.stats;

    std::cout << "\n=== Resumen Fase 0: carga y limpieza ===\n";
    std::cout << "Ruta activa: " << loadResult.filePath << "\n";
    std::cout << "Filas de datos leidas: " << stats.dataRowsRead << "\n";
    std::cout << "Filas almacenadas: " << stats.storedRows << "\n";
    std::cout << "Filas descartadas: " << stats.discardedRows << "\n";
    std::cout << "Filas cortas o mal formadas: " << stats.shortRows << "\n";

    std::cout << "\nValores ausentes detectados:\n";
    std::cout << "- TAIL_NUM: " << stats.missingTailNum << "\n";
    std::cout << "- ORIGIN_SEQ_ID: " << stats.missingOriginSeqId << "\n";
    std::cout << "- ORIGIN_AIRPORT: " << stats.missingOriginAirport << "\n";
    std::cout << "- DEST_SEQ_ID: " << stats.missingDestSeqId << "\n";
    std::cout << "- DEST_AIRPORT: " << stats.missingDestAirport << "\n";
    std::cout << "- DEP_TIME: " << stats.missingDepTime << "\n";
    std::cout << "- DEP_DELAY: " << stats.missingDepDelay << "\n";
    std::cout << "- ARR_TIME: " << stats.missingArrTime << "\n";
    std::cout << "- ARR_DELAY: " << stats.missingArrDelay << "\n";
    std::cout << "- WEATHER_DELAY: " << stats.missingWeatherDelay << "\n";

    std::cout << "\nCategorias detectadas:\n";
    std::cout << "- Aeropuertos unicos de origen: " << stats.uniqueOriginAirports << "\n";
    std::cout << "- Aeropuertos unicos de destino: " << stats.uniqueDestinationAirports << "\n";
}

/*
    printGpuSummary

    Informa del estado CUDA actual. Si no hay dispositivo disponible, imprime
    el motivo. Si lo hay, muestra propiedades basicas y, cuando el dataset ya
    esta cargado, una configuracion de lanzamiento sugerida para fases futuras.
*/
void printGpuSummary(const AppState& appState)
{
    std::cout << "\n=== Estado CUDA ===\n";

    if (!appState.deviceReady) {
        std::cout << "GPU no disponible: " << appState.deviceErrorMessage << "\n";
        return;
    }

    const unsigned long long globalMemoryInMb =
        static_cast<unsigned long long>(appState.deviceProp.totalGlobalMem) / (1024ULL * 1024ULL);
    const unsigned long long sharedMemoryPerBlockInKb =
        static_cast<unsigned long long>(appState.deviceProp.sharedMemPerBlock) / 1024ULL;

    std::cout << "Dispositivo: " << appState.deviceProp.name << "\n";
    std::cout << "Compute capability: " << appState.deviceProp.major
              << "." << appState.deviceProp.minor << "\n";
    std::cout << "Memoria global: " << globalMemoryInMb << " MB\n";
    std::cout << "Memoria compartida por bloque: " << sharedMemoryPerBlockInKb << " KB\n";
    std::cout << "Maximo de hilos por bloque: " << appState.deviceProp.maxThreadsPerBlock << "\n";

    if (appState.datasetLoaded) {
        const LaunchConfig launchConfig = computeLaunchConfig(
            static_cast<int>(getDatasetRowCount(appState.loadResult.dataset)),
            appState.deviceProp);

        std::cout << "Configuracion sugerida para el dataset actual: "
                  << launchConfig.blocks << " bloques x "
                  << launchConfig.threadsPerBlock << " hilos\n";
    }
}

/*
    printApplicationState

    Combina el resumen de carga y el resumen CUDA en una sola pantalla. Esta
    opcion del menu es util para depurar rapidamente el estado de la aplicacion
    sin tener que volver a cargar el dataset.
*/
void printApplicationState(const AppState& appState)
{
    std::cout << "\n========================================\n";
    std::cout << " Estado actual de la aplicacion\n";
    std::cout << "========================================\n";

    if (appState.datasetLoaded) {
        printLoadSummary(appState.loadResult);
    } else {
        std::cout << "\nDataset no cargado actualmente.\n";
    }

    printGpuSummary(appState);
}

/*
    loadDatasetIntoState

    Envuelve la llamada a loadDataset y, si tiene exito, actualiza el estado
    global de la aplicacion. Si falla, muestra el error y no marca el dataset
    como cargado.
*/
bool loadDatasetIntoState(AppState& appState, const std::string& datasetPath)
{
    CsvLoadResult newLoadResult = loadDataset(datasetPath);

    if (!newLoadResult.success) {
        std::cout << "\nNo se ha podido cargar el dataset.\n";
        std::cout << "Motivo: " << newLoadResult.errorMessage << "\n";
        std::cout << "Ruta probada: " << datasetPath << "\n";
        return false;
    }

    appState.datasetPath = datasetPath;
    appState.loadResult = std::move(newLoadResult);
    appState.datasetLoaded = true;

    printLoadSummary(appState.loadResult);
    return true;
}

/*
    promptAndLoadDataset

    Gestiona el ciclo completo de seleccion y carga del CSV:

    - pregunta la ruta;
    - permite cancelar si allowCancel es true;
    - intenta cargar el fichero;
    - repite el proceso si la carga falla.
*/
bool promptAndLoadDataset(AppState& appState, bool allowCancel)
{
    while (true) {
        const std::string selectedPath = promptDatasetPath(buildDatasetCandidates(appState));

        if (selectedPath.empty()) {
            if (allowCancel) {
                std::cout << "Operacion cancelada por el usuario.\n";
                return false;
            }

            std::cout << "Debe indicar una ruta valida para poder continuar.\n";
            continue;
        }

        if (loadDatasetIntoState(appState, selectedPath)) {
            return true;
        }

        std::cout << "Pruebe otra ruta o pulse X para cancelar la operacion.\n";
    }
}

/*
    printSuggestedLaunchConfigIfAvailable

    Muestra la configuracion sugerida solo cuando ya hay dataset cargado y
    tambien existe GPU CUDA disponible. Evita imprimir informacion parcial que
    podria confundir al usuario.
*/
void printSuggestedLaunchConfigIfAvailable(const AppState& appState)
{
    if (!appState.datasetLoaded || !appState.deviceReady) {
        return;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(
        static_cast<int>(getDatasetRowCount(appState.loadResult.dataset)),
        appState.deviceProp);

    std::cout << "- Configuracion sugerida actual: "
              << launchConfig.blocks << " bloques x "
              << launchConfig.threadsPerBlock << " hilos\n";
}

/*
    runPhase1Shell

    Submenu real de la Fase 01. Esta funcion:

    - comprueba que haya dataset y GPU;
    - pide tipo de filtro y umbral no negativo;
    - resume la configuracion elegida;
    - lanza la fase real en GPU.
*/
void runPhase1Shell(const AppState& appState)
{
    printPhase1Menu();

    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado. Cargue un CSV antes de continuar.\n";
        waitForEnter();
        return;
    }

    if (!appState.deviceReady) {
        std::cout << "No hay GPU CUDA disponible para ejecutar la Fase 01.\n";
        std::cout << "Motivo: " << appState.deviceErrorMessage << "\n";
        waitForEnter();
        return;
    }

    DelayFilterMode filterMode = DelayFilterMode::Both;
    int threshold = 0;

    if (!readDelayFilterModeOption("Seleccione el tipo de filtro (o X para volver): ", filterMode)) {
        return;
    }

    if (!readBoundedIntOption("Introduzca el umbral (>= 0, o X para volver): ", 0, 2147483647, threshold)) {
        return;
    }

    std::cout << "\nConfiguracion capturada:\n";
    std::cout << "- Columna objetivo: DEP_DELAY\n";
    std::cout << "- Tipo de filtro: " << getDelayFilterModeLabel(filterMode) << "\n";
    std::cout << "- Umbral: " << threshold << " minutos\n";
    printSuggestedLaunchConfigIfAvailable(appState);

    if (!runPhase1Computation(appState, filterMode, threshold)) {
        std::cout << "La Fase 01 no se ha podido completar correctamente.\n";
    }

    waitForEnter();
}

/*
    runPhase2Shell

    Submenu real de la Fase 02. Esta funcion recoge tipo de filtro y umbral,
    resume la configuracion y ejecuta el flujo host/device que pide el
    enunciado.
*/
void runPhase2Shell(const AppState& appState)
{
    printPhase2Menu();

    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado. Cargue un CSV antes de continuar.\n";
        waitForEnter();
        return;
    }

    if (!appState.deviceReady) {
        std::cout << "No hay GPU CUDA disponible para ejecutar la Fase 02.\n";
        std::cout << "Motivo: " << appState.deviceErrorMessage << "\n";
        waitForEnter();
        return;
    }

    DelayFilterMode filterMode = DelayFilterMode::Both;
    int threshold = 0;

    if (!readDelayFilterModeOption("Seleccione el tipo de filtro (o X para volver): ", filterMode)) {
        return;
    }

    if (!readBoundedIntOption("Introduzca el umbral (>= 0, o X para volver): ", 0, 2147483647, threshold)) {
        return;
    }

    std::cout << "\nConfiguracion capturada:\n";
    std::cout << "- Columna objetivo: ARR_DELAY\n";
    std::cout << "- Columna auxiliar: TAIL_NUM\n";
    std::cout << "- Tipo de filtro: " << getDelayFilterModeLabel(filterMode) << "\n";
    std::cout << "- Umbral: " << threshold << " minutos\n";
    printSuggestedLaunchConfigIfAvailable(appState);

    if (!runPhase2Computation(appState, filterMode, threshold)) {
        std::cout << "La Fase 02 no se ha podido completar correctamente.\n";
    }

    waitForEnter();
}

/*
    runPhase3Shell

    Prepara la interfaz de la Fase 03:

    - pide columna de trabajo;
    - pide tipo de reduccion;
    - resume la configuracion elegida;
    - recuerda las variantes que quedaran por implementar.
*/
void runPhase3Shell(const AppState& appState)
{
    printPhase3Menu();

    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado. Cargue un CSV antes de continuar.\n";
        waitForEnter();
        return;
    }

    int columnOption = 0;
    int reductionOption = 0;

    std::cout << "1. DEP_DELAY\n";
    std::cout << "2. ARR_DELAY\n";
    std::cout << "3. WEATHER_DELAY\n";

    if (!readBoundedIntOption("Seleccione la columna (1-3, o X para volver): ", 1, 3, columnOption)) {
        return;
    }

    std::cout << "1. Maximo\n";
    std::cout << "2. Minimo\n";

    if (!readBoundedIntOption("Seleccione el tipo de reduccion (1-2, o X para volver): ", 1, 2, reductionOption)) {
        return;
    }

    const char* selectedColumn = "DEP_DELAY";

    // Convertimos la opcion numerica en una etiqueta legible para el resumen.
    if (columnOption == static_cast<int>(Phase3ColumnOption::ArrivalDelay)) {
        selectedColumn = "ARR_DELAY";
    } else if (columnOption == static_cast<int>(Phase3ColumnOption::WeatherDelay)) {
        selectedColumn = "WEATHER_DELAY";
    }

    const char* selectedReduction =
        reductionOption == static_cast<int>(ReductionTypeOption::Maximum) ? "Maximo" : "Minimo";

    std::cout << "\nConfiguracion capturada:\n";
    std::cout << "- Columna objetivo: " << selectedColumn << "\n";
    std::cout << "- Reduccion: " << selectedReduction << "\n";
    std::cout << "- Variantes futuras: Simple, Basica, Intermedia y Patron de reduccion\n";
    printSuggestedLaunchConfigIfAvailable(appState);
    printPhasePendingMessage("Fase 03");
    waitForEnter();
}

/*
    runPhase4Shell

    Prepara la interfaz de la Fase 04:

    - pide si el histograma sera de origen o destino;
    - pide el umbral minimo;
    - resume la configuracion seleccionada;
    - deja indicada la estrategia prevista basada en IDs y no en strings GPU.
*/
void runPhase4Shell(const AppState& appState)
{
    printPhase4Menu();

    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado. Cargue un CSV antes de continuar.\n";
        waitForEnter();
        return;
    }

    int airportType = 0;
    int threshold = 0;

    std::cout << "1. Aeropuertos de origen\n";
    std::cout << "2. Aeropuertos de destino\n";

    if (!readBoundedIntOption("Seleccione el tipo de aeropuerto (1-2, o X para volver): ", 1, 2, airportType)) {
        return;
    }

    if (!readBoundedIntOption("Introduzca el umbral minimo de ocurrencias (>= 0, o X para volver): ", 0, 2147483647, threshold)) {
        return;
    }

    const char* airportLabel =
        airportType == static_cast<int>(HistogramAirportTypeOption::Origin) ? "ORIGIN" : "DEST";

    std::cout << "\nConfiguracion capturada:\n";
    std::cout << "- Tipo de aeropuerto: " << airportLabel << "\n";
    std::cout << "- Umbral minimo de ocurrencias: " << threshold << "\n";
    std::cout << "- Estrategia prevista: histograma por IDs de aeropuerto, no por strings directos\n";
    printSuggestedLaunchConfigIfAvailable(appState);
    printPhasePendingMessage("Fase 04");
    waitForEnter();
}

} // namespace

/*
    main

    Punto de entrada del programa. Orquesta el flujo completo actual:

    - prepara el estado global;
    - muestra informacion del hardware;
    - fuerza una carga inicial del dataset;
    - entra en un bucle de menu hasta que el usuario decide salir.
*/
int main()
{
    AppState appState;

    printApplicationBanner();

    // Primero detectamos la GPU para poder informar al usuario desde el inicio.
    appState.deviceReady = queryGpuInfo(appState.deviceProp, appState.deviceErrorMessage);
    printGpuSummary(appState);

    // Sin dataset no tiene sentido mantener la aplicacion abierta, porque
    // todas las fases dependen de la carga inicial del CSV.
    if (!promptAndLoadDataset(appState, true)) {
        std::cout << "Saliendo sin cargar dataset.\n";
        return 0;
    }

    // Pausa breve para que el usuario pueda leer el resumen inicial.
    waitForEnter();

    bool keepRunning = true;

    while (keepRunning) {
        printMainMenu();

        switch (readMainMenuOption()) {
        case MainMenuOption::Phase1:
            runPhase1Shell(appState);
            break;

        case MainMenuOption::Phase2:
            runPhase2Shell(appState);
            break;

        case MainMenuOption::Phase3:
            runPhase3Shell(appState);
            break;

        case MainMenuOption::Phase4:
            runPhase4Shell(appState);
            break;

        // Permite cambiar de fichero sin reiniciar el programa completo.
        case MainMenuOption::ReloadCsv:
            promptAndLoadDataset(appState, true);
            break;

        // Muestra el estado actual del dataset y del hardware.
        case MainMenuOption::ShowStatus:
            printApplicationState(appState);
            waitForEnter();
            break;

        // Sale del bucle principal y termina la aplicacion.
        case MainMenuOption::Exit:
            keepRunning = false;
            break;
        }
    }

    std::cout << "\nAplicacion finalizada.\n";
    return 0;
}
