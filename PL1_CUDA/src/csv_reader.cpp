#include "csv_reader.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>

/*
    csv_reader.cpp

    Este archivo implementa el flujo completo de Fase 0:

    1. abrir el fichero CSV;
    2. leer y validar la cabecera;
    3. recorrer el fichero fila a fila;
    4. limpiar y convertir cada campo necesario;
    5. almacenar los datos en una estructura por columnas;
    6. calcular estadisticas de faltantes y categorias unicas;
    7. devolver un resultado coherente que pueda mostrarse en la CLI.

    El codigo evita librerias externas a proposito para respetar el enunciado.
*/

namespace {

// Numero minimo de columnas que debe tener una fila valida del dataset.
// El CSV actual tiene mas control semantico, pero esta barrera evita acceder
// a posiciones que no existen en filas truncadas o corruptas.
constexpr std::size_t CSV_MIN_COLUMN_COUNT = 14;

// Indices exactos de las columnas relevantes dentro del dataset actual.
// Se mantienen como constantes para que sea facil defender de donde sale
// cada dato y para evitar numeros magicos dispersos por el codigo.
constexpr std::size_t IDX_TAIL_NUM = 3;
constexpr std::size_t IDX_ORIGIN_SEQ_ID = 5;
constexpr std::size_t IDX_ORIGIN_AIRPORT = 6;
constexpr std::size_t IDX_DEST_SEQ_ID = 7;
constexpr std::size_t IDX_DEST_AIRPORT = 8;
constexpr std::size_t IDX_DEP_TIME = 9;
constexpr std::size_t IDX_DEP_DELAY = 10;
constexpr std::size_t IDX_ARR_TIME = 11;
constexpr std::size_t IDX_ARR_DELAY = 12;
constexpr std::size_t IDX_WEATHER_DELAY = 13;

/*
    trimWhitespace

    Elimina espacios en blanco al principio y al final de un texto. Es un
    helper base porque la cabecera del CSV y muchos tokens pueden traer espacios
    o retornos de carro residuales que romperian validaciones posteriores.
*/
std::string trimWhitespace(const std::string& text)
{
    std::size_t start = 0;
    std::size_t end = text.size();

    // Avanza el inicio hasta el primer caracter no blanco.
    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    // Retrocede el final hasta el ultimo caracter no blanco.
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(start, end - start);
}

/*
    matchHeaderToken

    Comprueba que una posicion concreta de la cabecera coincide con el nombre
    de columna esperado. Se usa desde validateHeader para que la comparacion sea
    legible y concentrar aqui la limpieza previa del token.
*/
bool matchHeaderToken(const std::vector<std::string>& header, std::size_t index, const char* expectedToken)
{
    return index < header.size() && cleanQuotedToken(header[index]) == expectedToken;
}

/*
    countMissingFloat

    Actualiza el contador asociado a una columna numerica cuando el valor final
    ha quedado como NAN. Esto nos permite registrar faltantes sin duplicar la
    misma condicion una y otra vez en el bucle principal.
*/
void countMissingFloat(float value, std::size_t& missingCounter)
{
    if (std::isnan(value)) {
        ++missingCounter;
    }
}

/*
    areColumnsAligned

    Verifica que todas las columnas tienen exactamente la misma longitud.
    Esta comprobacion es critica porque el resto del programa asumira que el
    indice i apunta siempre a la misma fila logica en cualquier columna.
*/
bool areColumnsAligned(const DatasetColumns& dataset, std::size_t expectedRows)
{
    return dataset.depDelay.size() == expectedRows &&
        dataset.arrDelay.size() == expectedRows &&
        dataset.weatherDelay.size() == expectedRows &&
        dataset.depTime.size() == expectedRows &&
        dataset.arrTime.size() == expectedRows &&
        dataset.tailNum.size() == expectedRows &&
        dataset.originSeqId.size() == expectedRows &&
        dataset.destSeqId.size() == expectedRows &&
        dataset.originAirport.size() == expectedRows &&
        dataset.destAirport.size() == expectedRows;
}

} // namespace

/*
    splitCsvLineSimple

    Parser CSV simple orientado a este dataset. Recorre la linea caracter a
    caracter y decide si una coma separa campos reales o si forma parte del
    contenido porque estamos dentro de comillas.
*/
std::vector<std::string> splitCsvLineSimple(const std::string& line)
{
    std::vector<std::string> tokens;
    std::string currentToken;
    bool insideQuotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char currentChar = line[i];

        // Si encontramos comillas, alternamos el estado "dentro/fuera" de un
        // campo quoted. Si hay dos comillas seguidas dentro de un campo quoted,
        // las tratamos como una comilla escapada del propio dato.
        if (currentChar == '"') {
            if (insideQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                currentToken.push_back('"');
                ++i;
            } else {
                insideQuotes = !insideQuotes;
            }
            continue;
        }

        // Una coma solo separa columnas si no estamos dentro de comillas.
        if (currentChar == ',' && !insideQuotes) {
            tokens.push_back(currentToken);
            currentToken.clear();
            continue;
        }

        // Ignoramos '\r' para soportar finales de linea Windows sin contaminar
        // el contenido del token final.
        if (currentChar != '\r') {
            currentToken.push_back(currentChar);
        }
    }

    // El ultimo token queda pendiente al salir del bucle y hay que guardarlo.
    tokens.push_back(currentToken);
    return tokens;
}

/*
    cleanQuotedToken

    Aplica una limpieza textual minima pero suficiente:

    - recorta espacios externos;
    - elimina comillas envolventes simples;
    - vuelve a recortar por si quedaban espacios junto a las comillas.
*/
std::string cleanQuotedToken(const std::string& token)
{
    std::string cleanedToken = trimWhitespace(token);

    if (cleanedToken.size() >= 2 &&
        cleanedToken.front() == '"' &&
        cleanedToken.back() == '"') {
        cleanedToken = cleanedToken.substr(1, cleanedToken.size() - 2);
    }

    return trimWhitespace(cleanedToken);
}

/*
    parseFloatOrNan

    Interpreta un campo numerico del CSV como float. Si el token no representa
    un numero valido, devolvemos NAN en lugar de inventar un valor como 0. Esa
    decision es clave para no mezclar "dato realmente cero" con "dato ausente".
*/
float parseFloatOrNan(const std::string& token)
{
    const std::string cleanedToken = cleanQuotedToken(token);

    // Campo vacio: el enunciado recomienda guardar NAN.
    if (cleanedToken.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    try {
        std::size_t processedCharacters = 0;
        const float parsedValue = std::stof(cleanedToken, &processedCharacters);

        // Solo aceptamos el valor si todo el token ha sido consumido.
        if (processedCharacters != cleanedToken.size()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        return parsedValue;
    }
    catch (...) {
        // Cualquier conversion fallida se modela como dato ausente.
        return std::numeric_limits<float>::quiet_NaN();
    }
}

/*
    parseIntFromFloatToken

    El CSV trae algunos identificadores enteros escritos con formato decimal
    como "1129806.0". Esta funcion los convierte a entero truncado. Si el dato
    no existe o no es valido, devuelve false y deja -1 como centinela.
*/
bool parseIntFromFloatToken(const std::string& token, int& parsedValue)
{
    const float parsedFloat = parseFloatOrNan(token);

    if (std::isnan(parsedFloat)) {
        parsedValue = -1;
        return false;
    }

    parsedValue = static_cast<int>(parsedFloat);
    return true;
}

/*
    validateHeader

    Comprueba que el CSV tiene las columnas que necesitamos en las posiciones
    esperadas. No buscamos un parser generico: buscamos confirmar rapido que
    el fichero cargado es realmente el dataset de la practica.
*/
bool validateHeader(const std::vector<std::string>& header)
{
    if (header.size() < CSV_MIN_COLUMN_COUNT) {
        return false;
    }

    return matchHeaderToken(header, IDX_TAIL_NUM, "TAIL_NUM") &&
        matchHeaderToken(header, IDX_ORIGIN_SEQ_ID, "ORIGIN_SEQ_ID") &&
        matchHeaderToken(header, IDX_ORIGIN_AIRPORT, "ORIGIN_AIRPORT") &&
        matchHeaderToken(header, IDX_DEST_SEQ_ID, "DEST_SEQ_ID") &&
        matchHeaderToken(header, IDX_DEST_AIRPORT, "DEST_AIRPORT") &&
        matchHeaderToken(header, IDX_DEP_TIME, "DEP_TIME") &&
        matchHeaderToken(header, IDX_DEP_DELAY, "DEP_DELAY") &&
        matchHeaderToken(header, IDX_ARR_TIME, "ARR_TIME") &&
        matchHeaderToken(header, IDX_ARR_DELAY, "ARR_DELAY") &&
        matchHeaderToken(header, IDX_WEATHER_DELAY, "WEATHER_DELAY");
}

/*
    loadDataset

    Funcion principal de la Fase 0. Ejecuta de principio a fin la carga del
    fichero y devuelve un objeto con:

    - los datos ya limpios;
    - estadisticas de calidad;
    - mensaje de error si algo falla.
*/
CsvLoadResult loadDataset(const std::string& filename)
{
    CsvLoadResult result;
    result.filePath = filename;

    // Intentamos abrir el CSV en modo lectura simple de texto.
    std::ifstream file(filename.c_str());

    if (!file.is_open()) {
        result.errorMessage = "No se pudo abrir el archivo CSV indicado.";
        return result;
    }

    std::string headerLine;

    // La primera linea del fichero debe ser la cabecera del CSV.
    if (!std::getline(file, headerLine)) {
        result.errorMessage = "El archivo CSV esta vacio o no se pudo leer la cabecera.";
        return result;
    }

    // Parseamos la cabecera igual que cualquier fila y despues limpiamos cada
    // token para que la validacion no dependa de espacios residuales.
    result.header = splitCsvLineSimple(headerLine);

    for (std::size_t i = 0; i < result.header.size(); ++i) {
        result.header[i] = cleanQuotedToken(result.header[i]);
    }

    if (!validateHeader(result.header)) {
        result.errorMessage = "La cabecera del CSV no coincide con el formato esperado por la practica.";
        return result;
    }

    // Estos conjuntos sirven para medir el numero de aeropuertos distintos
    // detectados mientras recorremos el dataset una sola vez.
    std::unordered_set<std::string> uniqueOriginAirports;
    std::unordered_set<std::string> uniqueDestinationAirports;

    std::string line;
    DatasetColumns& dataset = result.dataset;
    CsvLoadStats& stats = result.stats;

    while (std::getline(file, line)) {
        ++stats.dataRowsRead;

        const std::vector<std::string> fields = splitCsvLineSimple(line);

        // Si la fila no llega al numero minimo de columnas, no podemos confiar
        // en sus indices y la descartamos como fila estructuralmente invalida.
        if (fields.size() < CSV_MIN_COLUMN_COUNT) {
            ++stats.discardedRows;
            ++stats.shortRows;
            continue;
        }

        // 1. TAIL_NUM: texto simple, cadena vacia si falta.
        const std::string tailNum = cleanQuotedToken(fields[IDX_TAIL_NUM]);
        if (tailNum.empty()) {
            ++stats.missingTailNum;
        }
        dataset.tailNum.push_back(tailNum);

        // 2. ORIGIN_SEQ_ID: entero procedente de un token decimal.
        int originSeqId = -1;
        if (!parseIntFromFloatToken(fields[IDX_ORIGIN_SEQ_ID], originSeqId)) {
            ++stats.missingOriginSeqId;
        }
        dataset.originSeqId.push_back(originSeqId);

        // 3. ORIGIN_AIRPORT: codigo textual del aeropuerto de origen.
        const std::string originAirport = cleanQuotedToken(fields[IDX_ORIGIN_AIRPORT]);
        if (originAirport.empty()) {
            ++stats.missingOriginAirport;
        } else {
            uniqueOriginAirports.insert(originAirport);
        }
        dataset.originAirport.push_back(originAirport);

        // 4. DEST_SEQ_ID: mismo criterio que con ORIGIN_SEQ_ID.
        int destSeqId = -1;
        if (!parseIntFromFloatToken(fields[IDX_DEST_SEQ_ID], destSeqId)) {
            ++stats.missingDestSeqId;
        }
        dataset.destSeqId.push_back(destSeqId);

        // 5. DEST_AIRPORT: codigo textual del aeropuerto de destino.
        const std::string destAirport = cleanQuotedToken(fields[IDX_DEST_AIRPORT]);
        if (destAirport.empty()) {
            ++stats.missingDestAirport;
        } else {
            uniqueDestinationAirports.insert(destAirport);
        }
        dataset.destAirport.push_back(destAirport);

        // 6. DEP_TIME: float o NAN si el dato no existe.
        const float depTime = parseFloatOrNan(fields[IDX_DEP_TIME]);
        countMissingFloat(depTime, stats.missingDepTime);
        dataset.depTime.push_back(depTime);

        // 7. DEP_DELAY: retraso de salida en minutos.
        const float depDelay = parseFloatOrNan(fields[IDX_DEP_DELAY]);
        countMissingFloat(depDelay, stats.missingDepDelay);
        dataset.depDelay.push_back(depDelay);

        // 8. ARR_TIME: hora de llegada.
        const float arrTime = parseFloatOrNan(fields[IDX_ARR_TIME]);
        countMissingFloat(arrTime, stats.missingArrTime);
        dataset.arrTime.push_back(arrTime);

        // 9. ARR_DELAY: retraso de llegada.
        const float arrDelay = parseFloatOrNan(fields[IDX_ARR_DELAY]);
        countMissingFloat(arrDelay, stats.missingArrDelay);
        dataset.arrDelay.push_back(arrDelay);

        // 10. WEATHER_DELAY: retraso atribuido al tiempo meteorologico.
        const float weatherDelay = parseFloatOrNan(fields[IDX_WEATHER_DELAY]);
        countMissingFloat(weatherDelay, stats.missingWeatherDelay);
        dataset.weatherDelay.push_back(weatherDelay);

        // Si hemos llegado aqui, la fila queda oficialmente almacenada.
        ++stats.storedRows;
    }

    // Guardamos el total de codigos unicos detectados durante la lectura.
    stats.uniqueOriginAirports = uniqueOriginAirports.size();
    stats.uniqueDestinationAirports = uniqueDestinationAirports.size();

    // No tiene sentido dar por valida una carga que no produce ni una fila.
    if (stats.storedRows == 0) {
        result.errorMessage = "El CSV se ha leido, pero no se ha podido almacenar ninguna fila valida.";
        return result;
    }

    // Verificacion final para blindar el contrato por columnas de Fase 0.
    if (!areColumnsAligned(dataset, stats.storedRows)) {
        result.errorMessage = "Las columnas del dataset no han quedado alineadas tras la limpieza.";
        return result;
    }

    result.success = true;
    return result;
}

/*
    getDatasetRowCount

    Devuelve el numero de filas almacenadas. Tomamos depDelay como referencia
    porque la comprobacion final de alineacion garantiza que todas las columnas
    tienen exactamente la misma longitud.
*/
std::size_t getDatasetRowCount(const DatasetColumns& dataset)
{
    return dataset.depDelay.size();
}
