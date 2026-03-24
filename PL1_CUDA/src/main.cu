// Compilar → Ctrl + Shift + B
// Ejecutar → Ctrl + F5
// Kernel (se ejecuta en GPU)

// main.cu
// Control principal del programa
// Llama a CSV reader y lanza kernels GPU

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits.h>

#include "csv_reader.h"
#include "kernels.cuh"

int main() {

    std::cout << "=== PL1 CUDA - Fase 03 (Variante 3.1 Simple) ===\n" << std::flush;

    // Ruta del dataset
    std::string path = "data/Airline_dataset.csv";

    // --------------------------------------------------
    // 1. Cargar CSV en CPU
    // --------------------------------------------------
    std::cout << "Antes de cargar CSV...\n" << std::flush;

    // Se invoca al lector CSV que procesa todo el dataset
    std::vector<FlightData> data = loadCSV(path);

    std::cout << "Después de cargar CSV...\n" << std::flush;

    std::cout << "Datos cargados: " << data.size() << std::endl;

    // Comprobación básica
    if (data.empty()) {
        std::cerr << "No hay datos para procesar\n";
        return -1;
    }

    // --------------------------------------------------
    // 2. Selección de columna
    // --------------------------------------------------
    std::cout << "\nSelecciona columna:\n";
    std::cout << "1. DEP_DELAY\n";
    std::cout << "2. ARR_DELAY\n";
    std::cout << "3. WEATHER_DELAY\n";

    int columnOption;
    std::cin >> columnOption;

    std::cout << "Columna seleccionada: " << columnOption << std::endl;

    // Vector que contendrá los datos seleccionados
    std::vector<int> h_data;

    // Se extrae la columna elegida
    for (auto& f : data) {
        if (columnOption == 1)
            h_data.push_back(f.dep_delay);
        else if (columnOption == 2)
            h_data.push_back(f.arr_delay);
        else if (columnOption == 3)
            h_data.push_back(f.weather_delay);
        else {
            std::cerr << "Opción inválida\n";
            return -1;
        }
    }

    // --------------------------------------------------
    // 3. Tipo de reducción
    // --------------------------------------------------
    std::cout << "\nTipo de reducción:\n";
    std::cout << "1. Max\n";
    std::cout << "2. Min\n";

    int type;
    std::cin >> type;

    std::cout << "Tipo seleccionado: " << type << std::endl;

    bool isMax = (type == 1);

    // --------------------------------------------------
    // 4. Memoria GPU
    // --------------------------------------------------

    // Punteros en GPU
    int* d_data;
    int* d_result;

    int n = h_data.size();

    // Reservamos memoria en GPU
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Copiamos datos desde CPU → GPU
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Inicialización del resultado según operación
    int init = isMax ? INT_MIN : INT_MAX;
    cudaMemcpy(d_result, &init, sizeof(int), cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // 5. Configuración de ejecución
    // --------------------------------------------------

    // Número de hilos por bloque (típico valor)
    int threads = 256;

    // Número de bloques necesario
    int blocks = (n + threads - 1) / threads;

    std::cout << "Lanzando kernel con "
        << blocks << " bloques y "
        << threads << " hilos por bloque\n";

    // Lanzamiento del kernel en GPU
    reductionSimple << <blocks, threads >> > (d_data, d_result, n, isMax);

    // Esperar a que termine la GPU
    cudaDeviceSynchronize();

    // --------------------------------------------------
    // 6. Resultado
    // --------------------------------------------------

    int result;

    // Copiamos resultado desde GPU → CPU
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\nResultado final: " << result << std::endl;

    // --------------------------------------------------
    // 7. Liberación de memoria
    // --------------------------------------------------

    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}