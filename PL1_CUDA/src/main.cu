//Compilar → Ctrl + Shift + B
//Ejecutar → Ctrl + F5
// Kernel (se ejecuta en GPU)

// main.cu
// Control principal del programa
// Llama a CSV reader y lanza kernels GPU

#include <iostream>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "csv_reader.h"

int main() {
    std::cout << "Inicio del programa\n";

    // ------------------------------
    // 1️⃣ Leer dataset CSV
    // ------------------------------
    std::string path = "../data/us_airline_dataset.csv";
    FlightData data = readCSV(path);

    std::cout << "Vuelos leídos: " << data.dep_delay.size() << "\n";

     size_t n = data.dep_delay.size();

    // ------------------------------
    // 2️⃣ Reservar memoria GPU para Fase 01
    // ------------------------------
    float* d_dep_delay;
    cudaMalloc((void**)&d_dep_delay, n * sizeof(float));

    // Copiar vector desde CPU a GPU
    cudaMemcpy(d_dep_delay, data.dep_delay.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // ------------------------------
    // 3️⃣ Definir kernel Fase 01
    // ------------------------------
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float umbral = 60; // ejemplo: retrasos mayores a 60 min

    depDelayKernel << <blocks, threadsPerBlock >> > (d_dep_delay, n, umbral);

    // Sincronizar GPU
    cudaDeviceSynchronize();

    // ------------------------------
    // 4️⃣ Liberar memoria
    // ------------------------------
    cudaFree(d_dep_delay);

    std::cout << "Fin de ejecución\n";
    return 0;
}