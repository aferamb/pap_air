#include "parte4.cuh"

#include <iostream>
#include <vector>

#include "comun.cuh"

namespace {

/** @brief Kernel para calcular el histograma parcial por bloque usando memoria compartida.
 * El histograma parcial de cada bloque se almacena en memoria global para su posterior fusión.
 * Se calcula un histograma de frecuencias de los índices densos de los aeropuertos, donde cada bin representa un aeropuerto específico.
 * @param denseIndices Índices densos de los elementos.
 * @param totalElements Número total de elementos.
 * @param totalBins Número total de bins.
 * @param partialHistograms Puntero al histograma parcial.
 */
__global__ void phase4SharedHistogramKernel(
    const int* denseIndices,
    int totalElements,
    int totalBins,
    unsigned int* partialHistograms)
{
    extern __shared__ unsigned int sharedHistogram[]; // Histograma parcial en memoria compartida

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;

    // Inicializar el histograma parcial en memoria compartida a cero
    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        sharedHistogram[bin] = 0;
    }

    __syncthreads();

    // Incrementar el conteo del bin correspondiente al índice denso del aeropuerto usando una operación atómica para evitar condiciones de carrera
    if (globalIndex < totalElements) {
        atomicAdd(&sharedHistogram[denseIndices[globalIndex]], 1U);
    }

    __syncthreads();

    unsigned int* blockPartialHistogram = partialHistograms + blockIdx.x * totalBins; // Puntero al histograma parcial de este bloque en memoria global

    // Copiar el histograma parcial del bloque desde memoria compartida a memoria global
    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        blockPartialHistogram[bin] = sharedHistogram[bin];
    }
}

/** @brief Kernel para fusionar los histogramas parciales en un histograma final.
 *  Fusion final de los histogramas parciales.
 * @param partialHistograms Puntero al array de histogramas parciales.
 * @param partialCount Número de histogramas parciales.
 * @param totalBins Número total de bins.
 * @param finalHistogram Puntero al histograma final.
 */
__global__ void phase4MergeHistogramKernel(
    const unsigned int* partialHistograms,
    int partialCount,
    int totalBins,
    unsigned int* finalHistogram)
{
    const int binIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (binIndex >= totalBins) {
        return;
    }

    unsigned int totalCount = 0;

    // Sumar los conteos de cada histograma parcial para el bin actual
    for (int partialIndex = 0; partialIndex < partialCount; ++partialIndex) {
        totalCount += partialHistograms[partialIndex * totalBins + binIndex];
    }

    finalHistogram[binIndex] = totalCount; // Almacenar el conteo total en el histograma final
}

/** @brief Imprime el histograma de la Fase 04.
 * Imprime el histograma de frecuencias de los aeropuertos con una representación visual usando caracteres '#'.
 * @param airportLabel Etiqueta del aeropuerto.
 * @param threshold Umbral para mostrar los aeropuertos.
 * @param histogram Histograma a imprimir.
 * @param denseToSeqId Mapeo de índices densos a IDs secuenciales.
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

    // Calcular el número de aeropuertos que cumplen con el umbral y el conteo máximo para la representación visual
    for (std::size_t i = 0; i < histogram.size(); ++i) {
        if (histogram[i] >= minimumCount) {
            ++shownAirports;

            if (histogram[i] > maximumShownCount) {
                maximumShownCount = histogram[i]; // Actualizar el conteo máximo para la representación visual
            }
        }
    }

    std::cout << "\n(4) Histograma de aeropuertos de " << airportLabel << "\n";
    std::cout << "Num de aeropuertos encontrados: " << denseToSeqId.size() << "\n\n";

    // Imprimir el histograma con una barra visual proporcional al conteo de vuelos para cada aeropuerto que cumple con el umbral
    for (std::size_t denseIndex = 0; denseIndex < histogram.size(); ++denseIndex) {
        // Omitir aeropuertos que no cumplen con el umbral de conteo
        if (histogram[denseIndex] < minimumCount) {
            continue;
        }

        const int seqId = denseToSeqId[denseIndex];
        // Buscar el código del aeropuerto usando el ID secuencial
        // const_iterator se usa para evitar modificaciones accidentales del mapa, aunque no es estrictamente necesario en este contexto de solo lectura
        const std::unordered_map<int, std::string>::const_iterator codeIt = idToCode.find(seqId); 
        const std::string airportCode = codeIt == idToCode.end() ? "" : codeIt->second; // Si no se encuentra el código, se muestra una cadena vacía

        std::cout << airportCode << " (" << seqId << ") | " << histogram[denseIndex] << " ";

        int barLength = 0;
        const int maxBarWidth = 40;

        // Calcular la longitud de la barra visual proporcional al conteo de vuelos, escalada al máximo para que el aeropuerto con más vuelos tenga una barra de longitud máxima
        if (maximumShownCount > 0) {
            barLength = static_cast<int>(
                (static_cast<unsigned long long>(histogram[denseIndex]) * static_cast<unsigned long long>(maxBarWidth)) /
                static_cast<unsigned long long>(maximumShownCount));
        }

        // Asegurar que los aeropuertos con conteos bajos pero que cumplen con el umbral tengan al menos una barra de longitud mínima para ser visibles en la representación visual
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

} // namespace

/** @brief Función principal de la Fase 04.
 * Esta función coordina la ejecución de los kernels para calcular el histograma de aeropuertos y luego imprime los resultados.
 * Selecciona la columna de aeropuertos a procesar (origen o destino) según el parámetro airportOption, y utiliza el umbral para filtrar los aeropuertos que se mostrarán en el histograma.
 * Primero, verifica que los datos sean válidos y que el histograma pueda caber en la memoria compartida de la GPU. 
 * Luego, calcula la configuración de lanzamiento para los kernels, ejecuta el kernel para calcular los histogramas parciales por bloque y el kernel para fusionar los histogramas parciales 
 * en un histograma final. Finalmente, copia el histograma final a la memoria host y llama a la función para imprimir el histograma.
 * 
 * @param airportOption Opción para seleccionar entre aeropuertos de origen (1) o destino (2).
 * @param threshold Umbral para mostrar los aeropuertos en el histograma.
 */
void phase04(int airportOption, int threshold)
{
    const bool useOrigin = airportOption == 1;      // Si es true, se procesan los aeropuertos de origen; si es false, se procesan los aeropuertos de destino
    const int totalElements = useOrigin ? g_originTotalElements : g_destinationTotalElements;       // Número total de elementos en la columna seleccionada (originSeqId o destSeqId)
    const int totalBins = useOrigin ? g_originTotalBins : g_destinationTotalBins;       // Número total de bins únicos en la columna seleccionada, que corresponde al número de aeropuertos únicos a procesar
    const int* denseInput = useOrigin ? d_originDenseInput : d_destinationDenseInput;       // Puntero a la columna de aeropuertos mapeada a IDs densos en la GPU para la columna seleccionada (originSeqId o destSeqId)
    const std::vector<int>& denseToSeqId = useOrigin ? g_originDenseToSeqId : g_destinationDenseToSeqId;        // Vector que mapea los IDs densos a los IDs originales para la columna seleccionada, utilizado para mostrar los códigos de aeropuerto en el histograma
    const std::unordered_map<int, std::string>& idToCode = useOrigin ? g_dataset.originIdToCode : g_dataset.destIdToCode;       // Mapa de IDs originales a códigos de aeropuerto para la columna seleccionada, utilizado para mostrar los códigos de aeropuerto en el histograma
    const char* airportLabel = useOrigin ? "origen" : "destino";        // Etiqueta para mostrar en el resumen de la GPU y en el título del histograma, indicando si se trata de aeropuertos de origen o destino

    if (totalElements <= 0 || totalBins <= 0 || denseInput == nullptr) {
        std::cout << "No hay datos validos para la Fase 04.\n";
        return;
    }

    const std::size_t sharedBytes = static_cast<std::size_t>(totalBins) * sizeof(unsigned int); // Cantidad de memoria compartida necesaria para almacenar el histograma parcial de un bloque, que depende del número de bins únicos (aeropuertos) a procesar

    // Verificar que el histograma parcial pueda caber en la memoria compartida de la GPU, ya que cada bloque necesita almacenar un histograma completo para su parte de los datos. 
    // Si el número de bins es demasiado grande, el histograma no cabrá en la memoria compartida y no se podrá ejecutar el kernel correctamente.
    if (sharedBytes > static_cast<std::size_t>(g_deviceProp.sharedMemPerBlock)) {
        std::cout << "El histograma no cabe en la memoria compartida por bloque de esta GPU.\n";
        return;
    }

    const LaunchConfig launchConfig = computeLaunchConfig(totalElements); // Calcular la configuración de lanzamiento para el kernel de histograma parcial en función del número total de elementos a procesar
    const LaunchConfig mergeLaunchConfig = computeLaunchConfig(totalBins); // Calcular la configuración de lanzamiento para el kernel de fusión de histogramas parciales en función del número total de bins

    std::cout << airportLabel
              << " | filas validas " << totalElements
              << " | bins " << totalBins
              << " | " << launchConfig.blocks << " x " << launchConfig.threadsPerBlock << "\n";

    unsigned int* devicePartialHistograms = nullptr;
    unsigned int* deviceFinalHistogram = nullptr;

    const std::size_t partialBytes =
        static_cast<std::size_t>(launchConfig.blocks) * static_cast<std::size_t>(totalBins) * sizeof(unsigned int);
    const std::size_t finalBytes = static_cast<std::size_t>(totalBins) * sizeof(unsigned int);

    cudaMalloc(reinterpret_cast<void**>(&devicePartialHistograms), partialBytes); // Reservar memoria en la GPU para los histogramas parciales de cada bloque, donde cada bloque tiene un histograma completo de bins
    cudaMalloc(reinterpret_cast<void**>(&deviceFinalHistogram), finalBytes); // Reservar memoria en la GPU para el histograma final fusionado, que tiene un conteo total para cada bin

    phase4SharedHistogramKernel<<<launchConfig.blocks, launchConfig.threadsPerBlock, sharedBytes>>>(
        denseInput,
        totalElements,
        totalBins,
        devicePartialHistograms);

    if (!executeAndWait("phase4SharedHistogramKernel")) { // Verificar si el kernel de histograma parcial se ejecutó correctamente, y si no, liberar la memoria reservada y mostrar un mensaje de error
        cudaFree(devicePartialHistograms);
        cudaFree(deviceFinalHistogram);
        std::cout << "La Fase 04 no se ha podido completar.\n";
        return;
    }

    phase4MergeHistogramKernel<<<mergeLaunchConfig.blocks, mergeLaunchConfig.threadsPerBlock>>>(
        devicePartialHistograms,
        launchConfig.blocks,
        totalBins,
        deviceFinalHistogram);

    if (!executeAndWait("phase4MergeHistogramKernel")) { // Verificar si el kernel de fusión de histogramas parciales se ejecutó correctamente, y si no, liberar la memoria reservada y mostrar un mensaje de error
        cudaFree(devicePartialHistograms); 
        cudaFree(deviceFinalHistogram);
        std::cout << "La Fase 04 no se ha podido completar.\n";
        return;
    }

    std::vector<unsigned int> histogram(static_cast<std::size_t>(totalBins), 0U);

    cudaMemcpy(histogram.data(), deviceFinalHistogram, finalBytes, cudaMemcpyDeviceToHost); // Copiar el histograma final desde la memoria global de la GPU a la memoria host para su posterior impresión

    cudaFree(devicePartialHistograms); // Be free
    cudaFree(deviceFinalHistogram); // Be free

    printPhase4Histogram(airportLabel, threshold, histogram, denseToSeqId, idToCode); // Imprimir el histograma de aeropuertos utilizando la función de impresión, mostrando solo los aeropuertos que cumplen con el umbral de conteo y utilizando los mapeos para mostrar los códigos de aeropuerto
}
