#include <cctype>
#include <climits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "comun.cuh"
#include "dataset_gpu.cuh"
#include "parte1.cuh"
#include "parte2.cuh"
#include "parte3.cuh"
#include "parte4.cuh"

/*
    main.cu

    Tras la modularizacion, este archivo se queda solo con la consola y el
    flujo principal. La logica real de cada fase vive en sus propios modulos.
*/

namespace {

std::string trimWhitespace(const std::string& text)
{
    std::size_t start = 0;
    std::size_t end = text.size();

    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(start, end - start);
}

bool fileExists(const std::string& path)
{
    std::ifstream file(path.c_str());
    return file.good();
}

bool isCancelToken(const std::string& input)
{
    return input == "x" || input == "X";
}

void pauseForEnter()
{
    std::cout << "\nPulse Intro para continuar...";
    std::string dummy;
    std::getline(std::cin, dummy);
}

bool readIntegerInRange(const char* prompt, const char* errorMessage, int minValue, int maxValue, int& value)
{
    while (true) {
        std::cout << prompt;

        std::string input;
        std::getline(std::cin, input);
        input = trimWhitespace(input);

        if (isCancelToken(input)) {
            return false;
        }

        std::stringstream parser(input);
        int parsedValue = 0;
        char trailingCharacter = '\0';

        if ((parser >> parsedValue) &&
            !(parser >> trailingCharacter) &&
            parsedValue >= minValue &&
            parsedValue <= maxValue) {
            value = parsedValue;
            return true;
        }

        std::cout << errorMessage << "\n";
    }
}

bool readSignedThreshold(const char* prompt, int& value)
{
    while (true) {
        std::cout << prompt;

        std::string input;
        std::getline(std::cin, input);
        input = trimWhitespace(input);

        if (isCancelToken(input)) {
            return false;
        }

        std::stringstream parser(input);
        int parsedValue = 0;
        char trailingCharacter = '\0';

        if ((parser >> parsedValue) && !(parser >> trailingCharacter)) {
            value = parsedValue;
            return true;
        }

        std::cout << "Debe introducir un numero entero, o X.\n";
    }
}

bool promptAndLoadDataset(bool allowCancel)
{
    while (true) {
        std::string defaultPath;

        if (!g_datasetPath.empty() && fileExists(g_datasetPath)) {
            defaultPath = g_datasetPath;
        } else if (fileExists("src/data/Airline_dataset.csv")) {
            defaultPath = "src/data/Airline_dataset.csv";
        }

        std::cout << "\nRuta del CSV";

        if (!defaultPath.empty()) {
            std::cout << " [Intro = " << defaultPath << "]";
        }

        std::cout << " [X = volver]\n";
        std::cout << "> ";

        std::string selectedPath;
        std::getline(std::cin, selectedPath);
        selectedPath = trimWhitespace(selectedPath);

        if (isCancelToken(selectedPath)) {
            if (allowCancel) {
                return false;
            }

            std::cout << "Debe indicar una ruta valida.\n";
            continue;
        }

        if (selectedPath.empty()) {
            selectedPath = defaultPath;
        }

        if (selectedPath.empty()) {
            std::cout << "Debe indicar una ruta valida.\n";
            continue;
        }

        if (cargarDataset(selectedPath)) {
            return true;
        }
    }
}

} // namespace

int main()
{
    std::cout << "========================================\n";
    std::cout << " PL1 CUDA - US Airline Dataset Toolkit\n";
    std::cout << "========================================\n";

    queryGpuInfo();
    printGpuSummary();

    if (!promptAndLoadDataset(true)) {
        liberarGPU();
        std::cout << "Saliendo sin cargar dataset.\n";
        return 0;
    }

    pauseForEnter();

    bool keepRunning = true;

    while (keepRunning) {
        std::string optionInput;

        while (true) {
            std::cout << "\nMenu principal\n";
            std::cout << "1. Fase 01 - Retraso en salida\n";
            std::cout << "2. Fase 02 - Retraso en llegada\n";
            std::cout << "3. Fase 03 - Reduccion de retraso\n";
            std::cout << "4. Fase 04 - Histograma de aeropuertos\n";
            std::cout << "R. Recargar CSV\n";
            std::cout << "I. Ver estado de la aplicacion\n";
            std::cout << "X. Salir\n";
            std::cout << "> ";

            std::getline(std::cin, optionInput);
            optionInput = trimWhitespace(optionInput);

            if (optionInput == "1" ||
                optionInput == "2" ||
                optionInput == "3" ||
                optionInput == "4" ||
                optionInput == "R" || optionInput == "r" ||
                optionInput == "I" || optionInput == "i" ||
                isCancelToken(optionInput)) {
                break;
            }

            std::cout << "Opcion no valida.\n";
        }

        if (optionInput == "1") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 01 - DEP_DELAY\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral firmado (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                continue;
            }

            if (!phase01(threshold)) {
                std::cout << "La Fase 01 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "2") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 02 - ARR_DELAY + TAIL_NUM\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral firmado (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                continue;
            }

            if (!phase02(threshold)) {
                std::cout << "La Fase 02 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "3") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 03 - Reduccion\n";
            std::cout << "1. DEP_DELAY  2. ARR_DELAY  3. WEATHER_DELAY\n";

            int columnValue = 0;
            int reductionValue = 0;

            if (!readIntegerInRange("Columna: ", "Debe introducir un numero entre 1 y 3, o X.", 1, 3, columnValue)) {
                continue;
            }

            std::cout << "1. Maximo  2. Minimo\n";

            if (!readIntegerInRange("Reduccion: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, reductionValue)) {
                continue;
            }

            if (!phase03(columnValue, reductionValue)) {
                std::cout << "La Fase 03 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "4") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 04 - Histograma de aeropuertos\n";
            std::cout << "1. Origen  2. Destino\n";

            int airportValue = 0;
            int threshold = 0;

            if (!readIntegerInRange("Tipo de aeropuerto: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, airportValue)) {
                continue;
            }

            if (!readIntegerInRange("Umbral minimo (>= 0, X para volver): ", "Debe introducir un numero mayor o igual que 0, o X.", 0, INT_MAX, threshold)) {
                continue;
            }

            if (!phase04(airportValue, threshold)) {
                std::cout << "La Fase 04 no se ha podido completar.\n";
            }

            pauseForEnter();
            continue;
        }

        if (optionInput == "R" || optionInput == "r") {
            promptAndLoadDataset(true);
            continue;
        }

        if (optionInput == "I" || optionInput == "i") {
            if (g_datasetLoaded) {
                printLoadSummary();
            } else {
                std::cout << "\nDataset no cargado.\n";
            }

            printGpuSummary();
            pauseForEnter();
            continue;
        }

        keepRunning = false;
    }

    liberarGPU();
    std::cout << "\nAplicacion finalizada.\n";
    return 0;
}
