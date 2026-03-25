#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

#include "cli_utils.h"
#include "csv_reader.h"

/*
    main.cu

    Este archivo coordina el estado actual del programa. Hoy la aplicacion
    implementa de forma real la Fase 0 y deja preparada la interfaz para el
    resto de fases de la practica.

    Flujo general actual:

    1. mostrar banner;
    2. detectar la GPU CUDA disponible;
    3. pedir la ruta del CSV y cargarlo;
    4. mostrar resumen de la limpieza y del hardware;
    5. entrar en un menu principal persistente;
    6. permitir navegar por las fases 1-4, aunque aun esten pendientes;
    7. permitir recargar el dataset y consultar el estado.

    Toda la logica de esta iteracion se mantiene en host. La GPU se consulta
    para conocer el hardware y dejar preparada la configuracion futura de los
    kernels, pero las fases 1-4 todavia no ejecutan su computo CUDA real.
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
    appendCandidateIfMissing(candidates, "data/Airline_dataset.csv");

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
    const CsvLoadResult newLoadResult = loadDataset(datasetPath);

    if (!newLoadResult.success) {
        std::cout << "\nNo se ha podido cargar el dataset.\n";
        std::cout << "Motivo: " << newLoadResult.errorMessage << "\n";
        std::cout << "Ruta probada: " << datasetPath << "\n";
        return false;
    }

    appState.datasetPath = datasetPath;
    appState.loadResult = newLoadResult;
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

    Submenu de la Fase 01 en el estado actual del proyecto. Todavia no lanza
    kernels, pero ya:

    - comprueba que haya dataset cargado;
    - pide el umbral firmado;
    - muestra la configuracion capturada;
    - deja constancia de la futura configuracion CUDA sugerida.
*/
void runPhase1Shell(const AppState& appState)
{
    printPhase1Menu();

    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado. Cargue un CSV antes de continuar.\n";
        waitForEnter();
        return;
    }

    int threshold = 0;

    if (!readSignedInt("Introduzca el umbral firmado (o X para volver): ", threshold)) {
        return;
    }

    std::cout << "\nConfiguracion capturada:\n";
    std::cout << "- Columna objetivo: DEP_DELAY\n";
    std::cout << "- Umbral: " << threshold << " minutos\n";
    printSuggestedLaunchConfigIfAvailable(appState);
    printPhasePendingMessage("Fase 01");
    waitForEnter();
}

/*
    runPhase2Shell

    Equivalente de interfaz para la Fase 02. Igual que en la Fase 01, hoy solo
    prepara la conversacion con el usuario y deja claro que la logica CUDA de
    deteccion por ARR_DELAY y TAIL_NUM aun no esta integrada.
*/
void runPhase2Shell(const AppState& appState)
{
    printPhase2Menu();

    if (!appState.datasetLoaded) {
        std::cout << "No hay dataset cargado. Cargue un CSV antes de continuar.\n";
        waitForEnter();
        return;
    }

    int threshold = 0;

    if (!readSignedInt("Introduzca el umbral firmado (o X para volver): ", threshold)) {
        return;
    }

    std::cout << "\nConfiguracion capturada:\n";
    std::cout << "- Columna objetivo: ARR_DELAY\n";
    std::cout << "- Columna auxiliar: TAIL_NUM\n";
    std::cout << "- Umbral: " << threshold << " minutos\n";
    printSuggestedLaunchConfigIfAvailable(appState);
    printPhasePendingMessage("Fase 02");
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
