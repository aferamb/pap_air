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

    - Interfaz de usuario en consola;
    - Navegacion por las distintas fases del programa;
    - Invocacion de las funciones correspondientes a cada fase.
*/

namespace {
/**
 * @brief   Elimina los espacios en blanco al inicio y al final de una cadena de texto, devolviendo una nueva cadena sin esos espacios.
 * 
 * @param text La cadena de texto de entrada.
 * @return std::string La cadena de texto sin espacios en blanco al inicio y al final.
 */
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

/**
 * @brief Verifica si un archivo existe en la ruta especificada.
 * 
 * @param path La ruta del archivo a verificar.
 * @return true Si el archivo existe y se puede abrir correctamente.
 * @return false Si el archivo no existe o no se puede abrir.
 */
bool fileExists(const std::string& path)
{
    std::ifstream file(path.c_str());
    return file.good();
}

/**
 * @brief Verifica si la cadena de texto ingresada por el usuario corresponde a un token de cancelación (X o x).
 * 
 * @param input La cadena de texto ingresada por el usuario.
 * @return true Si la cadena es "x" o "X", indicando que el usuario desea cancelar la operación actual.
 * @return false Si la cadena no es un token de cancelación, lo que indica que el usuario desea continuar con la operación.
 */
bool isCancelToken(const std::string& input)
{
    return input == "x" || input == "X";
}

/**
 * @brief Pausa la ejecución del programa y espera a que el usuario pulse la tecla Enter para continuar.
 */
void pauseForEnter()
{
    std::cout << "\nPulse Intro para continuar...";
    std::string dummy; // Jejej dummy 
    std::getline(std::cin, dummy);
}

/**
 * @brief Lee un número entero del usuario dentro de un rango específico, mostrando un mensaje de error si la entrada no es válida, y permite cancelar la operación ingresando "X".
 * 
 * @param prompt El mensaje que se muestra al usuario para solicitar la entrada.
 * @param errorMessage El mensaje que se muestra si la entrada no es válida.
 * @param minValue El valor mínimo permitido para la entrada.
 * @param maxValue El valor máximo permitido para la entrada.
 * @param value Una referencia a una variable donde se almacenará el valor entero válido ingresado por el usuario.
 * @return true Si el usuario ingresó un número entero válido dentro del rango especificado.
 * @return false Si el usuario ingresó "X" para cancelar la operación.
 */
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

        // Se intenta parsear la entrada como un número entero, verificando que no haya caracteres adicionales después del número, y que el número esté dentro del rango permitido. 
        //Si la entrada es válida, se asigna el valor a la variable de salida y se devuelve true. 
        //Si la entrada no es válida, se muestra un mensaje de error y se vuelve a solicitar la entrada al usuario.
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

/**
 * @brief Lee un número entero que puede ser positivo o negativo del usuario, mostrando un mensaje de error si la entrada no es válida, y permite cancelar la operación ingresando "X".
 * 
 * @param prompt El mensaje que se muestra al usuario para solicitar la entrada.
 * @param value Una referencia a una variable donde se almacenará el valor entero válido ingresado por el usuario.
 * @return true Si el usuario ingresó un número entero válido.
 * @return false Si el usuario ingresó "X" para cancelar la operación.
 */
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

/**
 * @brief Solicita al usuario que ingrese la ruta de un archivo CSV para cargar el dataset, ofreciendo una ruta por defecto si ya se ha cargado un dataset previamente o si existe un archivo en una ubicación común, y permite cancelar la operación ingresando "X".
 * 
 * @param allowCancel Un booleano que indica si se permite cancelar la operación. Si es true, el usuario puede ingresar "X" para salir sin cargar un dataset. Si es false, el usuario debe ingresar una ruta válida para continuar.
 * @return true Si el usuario ingresó una ruta válida y se cargó el dataset correctamente.
 * @return false Si el usuario ingresó "X" para cancelar la operación (solo si allowCancel es true).
 */
bool promptAndLoadDataset(bool allowCancel)
{
    while (true) {
        std::string defaultPath;

        // Si ya se ha cargado un dataset previamente y la ruta sigue siendo válida, se ofrece esa ruta como opción por defecto. 
        //Si no, se verifica si existe un archivo CSV en una ubicación común (src/data/Airline_dataset.csv) y se ofrece esa ruta como opción por defecto si el archivo existe.
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

/**
 * @brief Función principal del programa.
 * Esta función muestra un menú interactivo en la consola que permite al usuario cargar un dataset desde un archivo CSV, 
 * seleccionar diferentes fases de análisis de datos (retraso en salida, retraso en llegada, reducción de retraso, histograma de aeropuertos), recargar el dataset, ver el estado actual de la aplicación o salir del programa. 
 * También libera recursos de GPU al finalizar la aplicación. 
 * 
 * Be free my dear GPU.....
 * 
 * @return int El código de salida del programa.
 */
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

    pauseForEnter(); // Esperate un momento

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
        // Si seleccuiona 1 ejecurta fase1 
        if (optionInput == "1") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 01 - DEP_DELAY\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                continue;
            }

            phase01(threshold); // Ejecuta la fase 1 con el umbral ingresado por el usuario

            pauseForEnter(); //Esperate antes de mostrar el menu otra vez
            continue;
        }
        // Si selecciona 2 ejecuta fase2
        if (optionInput == "2") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 02 - ARR_DELAY + TAIL_NUM\n";

            int threshold = 0;

            if (!readSignedThreshold(
                    "Umbral (positivo=retraso, negativo=adelanto, X para volver): ",
                    threshold)) {
                continue;
            }

            phase02(threshold); // Ejecuta la fase 2 con el umbral ingresado por el usuario

            pauseForEnter();
            continue;
        }
        // Si selecciona 3 ejecuta fase3
        if (optionInput == "3") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 03 - Reduccion\n";
            std::cout << "1. DEP_DELAY  2. ARR_DELAY  3. WEATHER_DELAY\n";

            int columnValue = 0;
            int reductionValue = 0;

            // Se solicita al usuario que ingrese la columna a analizar (DEP_DELAY, ARR_DELAY o WEATHER_DELAY) y el tipo de reducción (máximo o mínimo)
            if (!readIntegerInRange("Columna: ", "Debe introducir un numero entre 1 y 3, o X.", 1, 3, columnValue)) {
                continue;
            }

            std::cout << "1. Maximo  2. Minimo\n";
            // Se solicita al usuario que ingrese el tipo de reducción (máximo o mínimo) para la columna seleccionada
            if (!readIntegerInRange("Reduccion: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, reductionValue)) {
                continue;
            }

            phase03(columnValue, reductionValue); /// Ejecuta la fase 3 con la columna y el tipo de reducción seleccionados por el usuario

            pauseForEnter();
            continue;
        }
        // Si selecciona 4 ejecuta fase4
        if (optionInput == "4") {
            if (!datasetListoParaGPU()) {
                continue;
            }

            std::cout << "\nFase 04 - Histograma de aeropuertos\n";
            std::cout << "1. Origen  2. Destino\n";

            int airportValue = 0;
            int threshold = 0;

            // Se solicita al usuario que ingrese el tipo de aeropuerto a analizar (origen o destino) y el umbral mínimo de vuelos para incluir un aeropuerto en el histograma
            if (!readIntegerInRange("Tipo de aeropuerto: ", "Debe introducir un numero entre 1 y 2, o X.", 1, 2, airportValue)) {
                continue;
            }
            // Se solicita al usuario que ingrese el umbral mínimo de vuelos para incluir un aeropuerto en el histograma, permitiendo ingresar un número entero positivo o negativo (donde los positivos representan retrasos y los negativos representan adelantos)
            if (!readIntegerInRange("Umbral minimo (>= 0, X para volver): ", "Debe introducir un numero mayor o igual que 0, o X.", 0, INT_MAX, threshold)) {
                continue;
            }

            phase04(airportValue, threshold); // Ejecuta la fase 4 con el tipo de aeropuerto y el umbral mínimo de vuelos seleccionados por el usuario

            pauseForEnter(); // Ya aprendisate a esperar, hazlo de nuevo :)
            continue;
        }

        if (optionInput == "R" || optionInput == "r") {
            promptAndLoadDataset(true); // Permite al usuario recargar el dataset desde un archivo CSV, ofreciendo la opción de cancelar la operación y volver al menú principal
            continue;
        }

        if (optionInput == "I" || optionInput == "i") {
            if (g_datasetLoaded) {
                printLoadSummary(); // Muestra un resumen del estado actual del dataset cargado en la GPU
            } else {
                std::cout << "\nDataset no cargado.\n";
            }

            printGpuSummary(); // Muestra un resumen del estado actual de la GPU, incluyendo información sobre el modelo, la memoria disponible y el uso actual
            pauseForEnter(); // Y vuelve a esperar, que esto se esta poniendo pesado de leer
            continue;
        }

        keepRunning = false;
    }

    liberarGPU(); // Be free my dear GPU.....
    std::cout << "\nAplicacion finalizada.\n";
    return 0;
}
