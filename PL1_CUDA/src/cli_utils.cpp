#include "cli_utils.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

/*
    cli_utils.cpp

    Implementa la capa de entrada y salida por consola del estado actual del
    proyecto. Todo lo que hay aqui trabaja en host y evita que main.cu tenga
    que mezclar validacion de texto con la logica general del programa.
*/

namespace {

/*
    trimWhitespace

    Normaliza una cadena eliminando espacios exteriores. Se usa antes de
    interpretar cualquier entrada del usuario para evitar errores por espacios.
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

/*
    fileExists

    Comprueba de forma sencilla si un fichero puede abrirse. Se usa para
    detectar rutas candidatas por defecto del CSV antes de preguntar al usuario.
*/
bool fileExists(const std::string& path)
{
    std::ifstream file(path.c_str());
    return file.good();
}

/*
    findFirstExistingCandidate

    Recorre la lista de rutas candidatas y devuelve la primera que realmente
    existe en disco. Si ninguna es valida, devuelve cadena vacia.
*/
std::string findFirstExistingCandidate(const std::vector<std::string>& candidatePaths)
{
    for (std::size_t i = 0; i < candidatePaths.size(); ++i) {
        if (!candidatePaths[i].empty() && fileExists(candidatePaths[i])) {
            return candidatePaths[i];
        }
    }

    return "";
}

/*
    readTrimmedLine

    Muestra un prompt y lee una linea completa. Despues recorta espacios para
    que el resto de validaciones trabajen sobre una entrada normalizada.
*/
std::string readTrimmedLine(const std::string& prompt)
{
    std::cout << prompt;

    std::string input;
    std::getline(std::cin, input);
    return trimWhitespace(input);
}

/*
    isCancelToken

    Centraliza la interpretacion del caracter de cancelacion. Asi todas las
    funciones de lectura comparten la misma convencion: escribir X o x vuelve.
*/
bool isCancelToken(const std::string& input)
{
    return input == "x" || input == "X";
}

} // namespace

/*
    printApplicationBanner

    Presenta la aplicacion y resume el alcance actual del proyecto para que el
    usuario vea desde el arranque que ya existen fases CUDA conectadas ademas
    de la carga inicial del CSV.
*/
void printApplicationBanner()
{
    std::cout << "========================================\n";
    std::cout << " PL1 CUDA - US Airline Dataset Toolkit\n";
    std::cout << " Fases 0-4: practica completa conectada\n";
    std::cout << "========================================\n";
}

/*
    printMainMenu

    Muestra el menu principal de navegacion. Este menu concentra:

    - acceso a las cuatro fases de la practica;
    - recarga del CSV;
    - visualizacion del estado actual;
    - salida limpia de la aplicacion.
*/
void printMainMenu()
{
    std::cout << "\nMenu principal\n";
    std::cout << "1. Fase 01 - Retraso en salida\n";
    std::cout << "2. Fase 02 - Retraso en llegada\n";
    std::cout << "3. Fase 03 - Reduccion de retraso\n";
    std::cout << "4. Fase 04 - Histograma de aeropuertos\n";
    std::cout << "R. Recargar CSV\n";
    std::cout << "I. Ver estado de la aplicacion\n";
    std::cout << "X. Salir\n";
}

// Cabecera del submenu de Fase 01. Situa al usuario antes de pedir el umbral
// y recuerda que el filtrado se ejecutara sobre DEP_DELAY en GPU.
void printPhase1Menu()
{
    std::cout << "\n=== Fase 01 - Retraso en despegues ===\n";
    std::cout << "Se trabajara sobre la columna DEP_DELAY.\n";
    std::cout << "Se pediran tipo de filtro y umbral no negativo.\n";
}

// Cabecera del submenu de Fase 02. Deja claro que la fase combina ARR_DELAY
// con TAIL_NUM y que devolvera resultados tambien al host.
void printPhase2Menu()
{
    std::cout << "\n=== Fase 02 - Retraso en aterrizajes ===\n";
    std::cout << "Se trabajara sobre ARR_DELAY y TAIL_NUM.\n";
    std::cout << "Se pediran tipo de filtro y umbral no negativo.\n";
}

// Cabecera del submenu de Fase 03.
void printPhase3Menu()
{
    std::cout << "\n=== Fase 03 - Reduccion de retraso ===\n";
    std::cout << "Se pedira columna y tipo de reduccion.\n";
}

// Cabecera del submenu de Fase 04.
void printPhase4Menu()
{
    std::cout << "\n=== Fase 04 - Histograma de aeropuertos ===\n";
    std::cout << "Se pedira origen/destino y umbral minimo.\n";
    std::cout << "La GPU trabajara con SEQ_ID y la CPU mostrara el codigo del aeropuerto.\n";
}

/*
    promptDatasetPath

    Flujo de trabajo:

    1. busca la primera ruta candidata existente;
    2. la ofrece como valor por defecto si la encuentra;
    3. permite cancelar con X;
    4. si el usuario pulsa Intro y existe una ruta valida, la devuelve.
*/
std::string promptDatasetPath(const std::vector<std::string>& candidatePaths)
{
    const std::string defaultPath = findFirstExistingCandidate(candidatePaths);

    std::cout << "\nIntroduzca la ruta del CSV del dataset.\n";
    std::cout << "Escriba X para cancelar";

    if (!defaultPath.empty()) {
        std::cout << " o pulse Intro para usar por defecto:\n";
        std::cout << defaultPath << "\n";
    } else {
        std::cout << ".\n";
        std::cout << "No se ha encontrado una ruta por defecto valida en el repositorio.\n";
    }

    const std::string input = readTrimmedLine("Ruta CSV: ");

    if (isCancelToken(input)) {
        return "";
    }

    // Si hay ruta por defecto y el usuario pulsa Intro, reutilizamos esa ruta.
    if (input.empty()) {
        return defaultPath;
    }

    return input;
}

/*
    readMainMenuOption

    No devuelve control hasta recibir una opcion valida del menu principal.
    Convierte una cadena del usuario en el enum que usa main para controlar
    el flujo sin depender de comparaciones de texto dispersas.
*/
MainMenuOption readMainMenuOption()
{
    while (true) {
        const std::string input = readTrimmedLine("Seleccione una opcion: ");

        if (input == "1") {
            return MainMenuOption::Phase1;
        }

        if (input == "2") {
            return MainMenuOption::Phase2;
        }

        if (input == "3") {
            return MainMenuOption::Phase3;
        }

        if (input == "4") {
            return MainMenuOption::Phase4;
        }

        if (input == "R" || input == "r") {
            return MainMenuOption::ReloadCsv;
        }

        if (input == "I" || input == "i") {
            return MainMenuOption::ShowStatus;
        }

        if (input == "X" || input == "x") {
            return MainMenuOption::Exit;
        }

        std::cout << "Opcion no valida. Intente de nuevo.\n";
    }
}

/*
    readDelayFilterModeOption

    Interpreta el modo de filtrado de Fase 01 y Fase 02. La decision de
    permitir Intro = Both sirve para ofrecer un camino rapido cuando el usuario
    quiere buscar tanto retrasos como adelantos sin elegir una sola rama.
*/
bool readDelayFilterModeOption(const std::string& prompt, DelayFilterMode& value)
{
    while (true) {
        std::cout << "1. Retraso\n";
        std::cout << "2. Adelanto\n";
        std::cout << "3. Ambos\n";
        std::cout << "Intro. Ambos\n";

        const std::string input = readTrimmedLine(prompt);

        if (isCancelToken(input)) {
            return false;
        }

        if (input.empty() || input == "3") {
            value = DelayFilterMode::Both;
            return true;
        }

        if (input == "1") {
            value = DelayFilterMode::Delay;
            return true;
        }

        if (input == "2") {
            value = DelayFilterMode::Advance;
            return true;
        }

        std::cout << "Debe introducir 1, 2, 3, Intro o X para volver.\n";
    }
}

/*
    readBoundedIntOption

    Lee un entero y ademas exige que el valor este dentro de un rango cerrado.
    Se usa en submenus donde solo existen opciones numericas concretas y
    conviene rechazar cualquier valor fuera de rango.
*/
bool readBoundedIntOption(const std::string& prompt, int minValue, int maxValue, int& value)
{
    while (true) {
        const std::string input = readTrimmedLine(prompt);

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

        std::cout << "Debe introducir un numero entre " << minValue
                  << " y " << maxValue
                  << ", o X para volver.\n";
    }
}

/*
    waitForEnter

    Introduce una pausa sencilla entre pantallas para que el usuario pueda leer
    mensajes de estado antes de volver al menu principal.
*/
void waitForEnter()
{
    std::cout << "\nPulse Intro para continuar...";

    std::string dummy;
    std::getline(std::cin, dummy);
}
