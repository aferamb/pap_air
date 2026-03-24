#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>

/*
    Función: loadCSV
    Lee el fichero CSV y extrae las columnas:
        - DEP_DELAY (columna 10)
        - ARR_DELAY (columna 12)
        - WEATHER_DELAY (columna 13)

    - Convierte valores a float y luego a int (truncado)
    - Ignora valores no válidos
*/

std::vector<FlightData> loadCSV(const std::string& filename) {

    std::vector<FlightData> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error abriendo archivo: " << filename << std::endl;
        return data;
    }

    std::cout << "Archivo abierto correctamente\n";

    std::string line;

    // Leer cabecera
    std::getline(file, line);
    std::cout << "Cabecera: " << line << std::endl;

    int line_count = 0;

    while (std::getline(file, line)) {

        line_count++;
        // Mostrar progreso cada cierto número de líneas
        if (line_count % 100000 == 0) {
            std::cout << "Procesadas " << line_count << " lineas...\n" << std::flush;
        }

        std::stringstream ss(line);
        std::string value;

        FlightData flight{ 0, 0, 0 };
        bool valid = false;

        int col = 0;

        while (std::getline(ss, value, ',')) {

            try {
                // Limpiar comillas si existen
                if (!value.empty() && value.front() == '"')
                    value.erase(0, 1);

                if (!value.empty() && value.back() == '"')
                    value.pop_back();

                int val = static_cast<int>(std::stof(value));

                // Índices según dataset
                if (col == 10) { flight.dep_delay = val; valid = true; }
                if (col == 12) { flight.arr_delay = val; valid = true; }
                if (col == 13) {
                    flight.weather_delay = val; valid = true;

                }

            }
            catch (...) {
                // Ignorar valores no numéricos
            }

            col++;
        }

        // Solo guardamos si hay datos válidos
        if (valid) {
            data.push_back(flight);
        }
    }

    std::cout << "Total líneas procesadas: " << data.size() << std::endl;

    return data;
}