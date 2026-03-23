#pragma once
#include <vector>
#include <string>
#include <cmath> // para NAN

struct FlightData {
    std::vector<float> dep_delay;
    std::vector<float> arr_delay;
    std::vector<float> weather_delay;
    std::vector<std::string> tail_num;
};

// Declaración de la función
FlightData readCSV(const std::string& filename);