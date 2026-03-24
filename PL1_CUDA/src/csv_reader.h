#pragma once
#include <vector>
#include <string>

// Estructura que almacena los campos relevantes del dataset
struct FlightData {
    int dep_delay;
    int arr_delay;
    int weather_delay;
};

// Función que carga el CSV completo
std::vector<FlightData> loadCSV(const std::string& filename);