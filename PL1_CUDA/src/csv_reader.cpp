// csv_reader.cpp
// Funciones para leer el dataset CSV
// Cada columna relevante se guarda en un vector unidimensional
// Valores faltantes se representan con NAN

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath> // para NAN

// Estructura para almacenar columnas relevantes del dataset
struct FlightData {
    std::vector<float> dep_delay;     // Retraso en despegue
    std::vector<float> arr_delay;     // Retraso en aterrizaje
    std::vector<float> weather_delay; // Retraso por clima
    std::vector<std::string> tail_num;// Matrícula del avión
};

// Función para leer CSV y devolver FlightData
FlightData readCSV(const std::string& filename) {
    FlightData data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el CSV\n";
        return data;
    }

    // Saltar la primera línea (cabecera)
    std::getline(file, line);

    // Leer línea por línea
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        // Columna DEP_DELAY
        std::getline(ss, item, ',');
        try {
            data.dep_delay.push_back(std::stof(item));
        }
        catch (...) { data.dep_delay.push_back(NAN); }

        // Columna ARR_DELAY
        std::getline(ss, item, ',');
        try {
            data.arr_delay.push_back(std::stof(item));
        }
        catch (...) { data.arr_delay.push_back(NAN); }

        // Columna WEATHER_DELAY
        std::getline(ss, item, ',');
        try {
            data.weather_delay.push_back(std::stof(item));
        }
        catch (...) { data.weather_delay.push_back(NAN); }

        // Columna TAIL_NUM
        std::getline(ss, item, ',');
        data.tail_num.push_back(item);
    }

    return data;
}