#pragma once

#include <string>
#include <vector>

/*
    cli_utils.h

    Este modulo encapsula toda la interaccion por consola que ya existe en la
    aplicacion actual. Su objetivo es mantener el codigo de menu y validacion
    separado del lector CSV y separado tambien de la logica principal de main.
*/

// Opciones de alto nivel del menu principal. Main utiliza este enum para
// convertir texto introducido por el usuario en ramas de control claras.
enum class MainMenuOption {
    Phase1,
    Phase2,
    Phase3,
    Phase4,
    ReloadCsv,
    ShowStatus,
    Exit
};

// Columnas que el usuario puede elegir en la Fase 03 ya implementada.
enum class Phase3ColumnOption {
    DepartureDelay = 1,
    ArrivalDelay = 2,
    WeatherDelay = 3
};

// Tipo de reduccion que la Fase 03 aplica sobre la columna elegida.
enum class ReductionTypeOption {
    Maximum = 1,
    Minimum = 2
};

// Tipo de aeropuerto que se solicitara en la Fase 04.
enum class HistogramAirportTypeOption {
    Origin = 1,
    Destination = 2
};

// Tipo de filtro opcional para las Fases 01 y 02. Permite separar el concepto
// de "modo de busqueda" del valor numerico del umbral.
enum class DelayFilterMode {
    Delay = 1,
    Advance = 2,
    Both = 3
};

// Muestra el encabezado general del programa al arrancar la aplicacion.
void printApplicationBanner();

// Imprime el menu principal con todas las opciones disponibles hoy.
void printMainMenu();

// Imprime una introduccion breve para cada submenu de fase.
void printPhase1Menu();
void printPhase2Menu();
void printPhase3Menu();
void printPhase4Menu();

// Muestra el mensaje estandar utilizado cuando una fase aun no tiene logica
// CUDA conectada pero la interfaz ya recoge sus parametros.
void printPhasePendingMessage(const std::string& phaseName);

/*
    promptDatasetPath

    Pide la ruta del CSV al usuario. Si alguna ruta candidata existe en disco,
    la ofrece como valor por defecto al pulsar Intro. Si el usuario escribe X,
    la funcion devuelve una cadena vacia para indicar cancelacion.
*/
std::string promptDatasetPath(const std::vector<std::string>& candidatePaths);

// Lee la opcion del menu principal y no devuelve hasta tener una entrada valida.
MainMenuOption readMainMenuOption();

/*
    readSignedInt

    Lee un entero firmado desde consola. Si el usuario escribe X, devuelve
    false para que el llamador pueda volver al menu anterior sin error.
*/
bool readSignedInt(const std::string& prompt, int& value);

/*
    readDelayFilterModeOption

    Lee el modo de filtrado de las Fases 01 y 02. Si el usuario pulsa Intro,
    se interpreta como "ambos" para seguir la preferencia pedida en el flujo.
    Si el usuario escribe X, devuelve false para volver al menu anterior.
*/
bool readDelayFilterModeOption(const std::string& prompt, DelayFilterMode& value);

/*
    readBoundedIntOption

    Lee una opcion entera dentro de un rango cerrado. Es util para menus
    numericos donde no basta con parsear un entero: tambien hay que asegurar
    que pertenece a las opciones que la interfaz ofrece realmente.
*/
bool readBoundedIntOption(const std::string& prompt, int minValue, int maxValue, int& value);

// Pausa la ejecucion hasta que el usuario pulse Intro para continuar.
void waitForEnter();
