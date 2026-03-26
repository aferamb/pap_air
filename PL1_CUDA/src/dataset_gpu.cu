#include "dataset_gpu.cuh"

#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "comun.cuh"

namespace {

/*
    buildTailBuffer

    Convierte el vector de matriculas de longitud variable en un buffer
    linealizado de celdas fijas. Asi cada hilo de la Fase 02 puede recuperar su
    matricula con un simple desplazamiento.
*/
void buildTailBuffer(const std::vector<std::string>& source, std::vector<char>& outBuffer)
{
    outBuffer.assign(source.size() * kPhase2TailNumStride, '\0'); // Inicializa el buffer con ceros para asegurar que las celdas de matricula sin datos sean cadenas vacias

    // Copia cada matricula al buffer, truncando si es necesario para ajustarse a la longitud fija de cada celda
    for (std::size_t row = 0; row < source.size(); ++row) {
        const std::string& tailNum = source[row]; // Obtiene la matricula de la fila actual
        char* cell = &outBuffer[row * kPhase2TailNumStride]; // Calcula el puntero a la celda correspondiente en el buffer
        const std::size_t maxCharacters = static_cast<std::size_t>(kPhase2TailNumStride - 1); // Deja espacio para el caracter nulo al final de la celda
        const std::size_t charactersToCopy = tailNum.size() < maxCharacters ? tailNum.size() : maxCharacters; // Asegura que no se copie mas alla del limite de la celda

        for (std::size_t i = 0; i < charactersToCopy; ++i) {
            cell[i] = tailNum[i]; // Copia cada caracter de la matricula a la celda correspondiente en el buffer
        }

        cell[charactersToCopy] = '\0'; 
    }
}

/*
    buildDenseInput

    Transforma los SEQ_ID de aeropuerto en bins densos consecutivos para que el
    histograma de la Fase 04 no dependa del rango bruto de IDs.
*/
void buildDenseInput(
    const std::vector<int>& seqIds, // Vector de SEQ_IDs de origen o destino
    const std::unordered_map<int, std::string>& idToCode, // Mapa que relaciona SEQ_IDs con codigos de aeropuerto
    std::vector<int>& denseToSeqId, // Vector que mapea índices densos a SEQ_IDs
    std::vector<int>& denseInput) // Vector con los índices densos para cada fila
{
    denseToSeqId.clear();
    denseInput.clear();
    denseInput.reserve(seqIds.size());

    std::unordered_map<int, int> denseIndexBySeqId; // Mapa que relaciona SEQ_IDs con índices densos

    for (std::size_t i = 0; i < seqIds.size(); ++i) {
        const int seqId = seqIds[i];

        if (seqId < 0 || idToCode.find(seqId) == idToCode.end()) {
            continue;
        }

        const std::unordered_map<int, int>::const_iterator found = denseIndexBySeqId.find(seqId); // Busca el SEQ_ID en el mapa de índices densos

        // Si el SEQ_ID no se ha visto antes, asigna un nuevo índice denso y actualiza las estructuras correspondientes
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

/**
 * @brief Sube un dataset a la GPU, reservando memoria para las columnas de retrasos, matriculas, conteo y resultados de la Fase 02, y para los inputs densos de origen y destino si no están vacíos.
 * 
 * @param dataset 
 * @param errorMessage 
 * @return true Si el dataset se subió correctamente a GPU, false en caso contrario. En caso de error,
 */

bool subirDatasetAGPU(const DatasetColumns& dataset, std::string& errorMessage)
{
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

    const std::size_t delayBytes = static_cast<std::size_t>(rowCount) * sizeof(float);                          // El tamaño de las columnas de retrasos se calcula como el número de filas multiplicado por el tamaño de un float
    const std::size_t tailBytes = tailBuffer.size() * sizeof(char);                                             // El tamaño de la columna de matriculas se calcula como el tamaño total del buffer de matriculas multiplicado por el tamaño de un char
    const std::size_t outDelayBytes = static_cast<std::size_t>(rowCount) * sizeof(int);                         // El tamaño de la columna de retrasos de salida que cumplen la condición se calcula como el número de filas multiplicado por el tamaño de un int, ya que se almacenan como valores numéricos
    const std::size_t outTailBytes = static_cast<std::size_t>(rowCount) * kPhase2TailNumStride * sizeof(char);  // El tamaño de la columna de matriculas que cumplen la condición se calcula como el número de filas multiplicado por la longitud fija de cada celda de matricula y por el tamaño de un char

    // Reserva memoria en GPU para las columnas de retrasos, matriculas, conteo y resultados de la Fase 02, y para los inputs densos de origen y destino si no están vacíos
    cudaMalloc(reinterpret_cast<void**>(&d_depDelay), delayBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_arrDelay), delayBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_tailNums), tailBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_phase2Count), sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_phase2OutDelayValues), outDelayBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_phase2OutTailNums), outTailBytes);

    // Solo reserva memoria para los inputs densos si no están vacíos, para evitar reservar memoria innecesaria en GPU
    if (!originDenseInput.empty()) {
        cudaMalloc(
            reinterpret_cast<void**>(&d_originDenseInput),
            static_cast<std::size_t>(originDenseInput.size()) * sizeof(int));
    }

    // Solo reserva memoria para los inputs densos si no están vacíos, para evitar reservar memoria innecesaria en GPU
    if (!destinationDenseInput.empty()) {
        cudaMalloc(
            reinterpret_cast<void**>(&d_destinationDenseInput),
            static_cast<std::size_t>(destinationDenseInput.size()) * sizeof(int));
    }

    // Copia los datos de las columnas de retrasos, matriculas, y los inputs densos a GPU
    cudaMemcpy(d_depDelay, dataset.depDelay.data(), delayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrDelay, dataset.arrDelay.data(), delayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tailNums, tailBuffer.data(), tailBytes, cudaMemcpyHostToDevice);

    // Copia los datos de los inputs densos solo si no están vacíos, para evitar copiar datos innecesarios a GPU
    if (!originDenseInput.empty()) {
        cudaMemcpy(
            d_originDenseInput,
            originDenseInput.data(),
            static_cast<std::size_t>(originDenseInput.size()) * sizeof(int),
            cudaMemcpyHostToDevice);
    }
    // Copia los datos de los inputs densos solo si no están vacíos, para evitar copiar datos innecesarios a GPU
    if (!destinationDenseInput.empty()) {
        cudaMemcpy(
            d_destinationDenseInput,
            destinationDenseInput.data(),
            static_cast<std::size_t>(destinationDenseInput.size()) * sizeof(int),
            cudaMemcpyHostToDevice);
    }

    // Almacena en variables globales la información del dataset que se muestra en el resumen de la GPU y que se utiliza para calcular la configuración de lanzamiento base
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

} // namespace

/**
 * @brief Libera la memoria asignada en la GPU para las columnas de retrasos, matriculas, conteo y resultados de la Fase 02, y para los inputs densos de origen y destino. 
 * Tambien reinicia las variables globales,  no hay dataset cargado en GPU. 
 * Esta función se llama antes de subir un nuevo dataset a GPU para asegurar que no haya fugas de memoria y que el estado global sea consistente.
 * 
 */
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

/*
Muestra resumen de carga de dataset  con dotas de filas almaxcenada, valores ausentes y aeropuertos unicos. 
Esta función se llama después de cargar un dataset para proporcionar info 
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

    std::cout << "\nAeropuertos unicos detectados:\n";
    std::cout << "- Aeropuertos unicos de origen por SEQ_ID: " << g_summary.uniqueOriginSeqIds << "\n";
    std::cout << "- Aeropuertos unicos de destino por SEQ_ID: " << g_summary.uniqueDestinationSeqIds << "\n";
}

/** @brief Carga un dataset desde un archivo y lo prepara para su uso en la GPU.
 * @param datasetPath Ruta al archivo del dataset.
 * @return true si el dataset se cargó correctamente, false en caso contrario.
 */
bool cargarDataset(const std::string& datasetPath)
{
    DatasetColumns newDataset;
    LoadSummary newSummary;
    std::string errorMessage;

    // Intenta cargar el dataset desde el archivo especificado
    if (!loadDataset(datasetPath, newDataset, newSummary, errorMessage)) {
        std::cout << "No se ha podido cargar el dataset.\n";
        std::cout << "Motivo: " << errorMessage << "\n";
        return false;
    }

    // Si la GPU está lista, intenta subir el dataset a GPU. Si ocurre un error durante la subida, se muestra un mensaje de error y se devuelve false
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

/** @brief Verifica si el dataset está listo para ser utilizado en la GPU, comprobando que el dataset esté cargado, que la GPU esté disponible y que el dataset tenga filas válidas.
 * @return true si el dataset está listo para GPU, false en caso contrario. En caso de no estar listo, se muestra un mensaje indicando el motivo.
 */
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
