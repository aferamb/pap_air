#include "parte4.cuh"

#include <iostream>
#include <vector>

#include "comun.cuh"

namespace {

/*
    Histograma parcial por bloque usando memoria compartida.
*/
__global__ void phase4SharedHistogramKernel(
    const int* denseIndices,
    int totalElements,
    int totalBins,
    unsigned int* partialHistograms)
{
    extern __shared__ unsigned int sharedHistogram[];

    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = threadIdx.x;

    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        sharedHistogram[bin] = 0;
    }

    __syncthreads();

    if (globalIndex < totalElements) {
        atomicAdd(&sharedHistogram[denseIndices[globalIndex]], 1U);
    }

    __syncthreads();

    unsigned int* blockPartialHistogram = partialHistograms + blockIdx.x * totalBins;

    for (int bin = localIndex; bin < totalBins; bin += blockDim.x) {
        blockPartialHistogram[bin] = sharedHistogram[bin];
    }
}

/*
    Fusion final de los histogramas parciales.
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

    for (int partialIndex = 0; partialIndex < partialCount; ++partialIndex) {
        totalCount += partialHistograms[partialIndex * totalBins + binIndex];
    }

    finalHistogram[binIndex] = totalCount;
}

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
        const std::unordered_map<int, std::string>::const_iterator codeIt = idToCode.find(seqId);
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

} // namespace

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
