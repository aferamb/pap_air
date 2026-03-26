#include "csv_reader.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>

/*
    csv_reader.cpp
    

    Funciones para cargar el dataset desde un archivo CSV, incluyendo:
    - splitCsvLineSimple: Función para dividir una línea CSV en tokens, manejando correctamente las comillas y los espacios.
    - cleanQuotedToken: Función para limpiar un token que puede estar entre comillas, eliminando las comillas y los espacios innecesarios.
    - parseFloatOrNan: Función para convertir un token a float, devolviendo NaN si el token está vacío o no es un número válido.
    - parseIntFromFloatToken: Función para convertir un token a int, tratando el token como un float primero y devolviendo false si no se pudo convertir.
    - loadDataset: Función principal para cargar el dataset desde un archivo CSV, llenando la estructura DatasetColumns y actualizando el LoadSummary 
    con estadísticas sobre la carga. Maneja errores de apertura de archivo, lectura de líneas y formato de datos, y devuelve mensajes de error descriptivos cuando sea necesario.
    - trimWhitespace: Función auxiliar para eliminar espacios en blanco al inicio y al final de un string.
*/

namespace {
// Índices de las columnas relevantes en el CSV, definidos como constantes para facilitar su uso y mantenimiento
constexpr std::size_t CSV_MIN_COLUMN_COUNT = 14;
constexpr std::size_t IDX_TAIL_NUM = 3;
constexpr std::size_t IDX_ORIGIN_SEQ_ID = 5;
constexpr std::size_t IDX_ORIGIN_AIRPORT = 6;
constexpr std::size_t IDX_DEST_SEQ_ID = 7;
constexpr std::size_t IDX_DEST_AIRPORT = 8;
constexpr std::size_t IDX_DEP_DELAY = 10;
constexpr std::size_t IDX_ARR_DELAY = 12;
constexpr std::size_t IDX_WEATHER_DELAY = 13;

/** 
    trimWhitespace: Función auxiliar para eliminar espacios en blanco al inicio y al final de un string.
    La función utiliza dos índices, start y end, para identificar el rango del string que no contiene espacios. 
     Se incrementa start hasta encontrar un carácter no blanco y se decrementa end hasta encontrar un carácter no blanco, luego se devuelve la subcadena correspondiente a ese rango.
    @param text El string del cual se desea eliminar los espacios en blanco.
    @return Un nuevo string con los espacios en blanco eliminados del inicio y del final.
    
*/
std::string trimWhitespace(const std::string& text)
{
    std::size_t start = 0;
    std::size_t end = text.size();
    // Eliminar espacios en blanco al inicio
    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }
    // Eliminar espacios en blanco al final
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(start, end - start);
}

/**
 * @brief Limpia los datos del dataset borrando el contenido de todas las columnas y mapas. 
 * Esto se utiliza para reiniciar el estado del dataset antes de cargar nuevos datos, asegurando que no queden residuos de cargas anteriores que puedan afectar los resultados.
 * 
 * @param dataset 
 */
void clearDataset(DatasetColumns& dataset)
{
    dataset.depDelay.clear();
    dataset.arrDelay.clear();
    dataset.weatherDelay.clear();
    dataset.tailNum.clear();
    dataset.originSeqId.clear();
    dataset.destSeqId.clear();
    dataset.originIdToCode.clear();
    dataset.destIdToCode.clear();
}

/**
 * @brief Cuenta los valores float que son NaN.
 * 
 * @param value El valor float a comprobar.
 * @param counter Referencia al contador para incrementar si el valor es NaN.
 */
void countMissingFloat(float value, std::size_t& counter)
{
    if (std::isnan(value)) {
        ++counter;
    }
}

/**
 * @brief Recuerda el código de un aeropuerto asociado a un ID de secuencia.
 * 
 * @param idToCode Mapa que asocia IDs de secuencia con códigos de aeropuertos.
 * @param seqId ID de secuencia del aeropuerto.
 * @param code Código del aeropuerto.
 */
void rememberAirportCode(std::unordered_map<int, std::string>& idToCode, int seqId, const std::string& code)
{
    if (seqId >= 0 && !code.empty() && idToCode.find(seqId) == idToCode.end()) {
        idToCode[seqId] = code;
    }
}

} // namespace

/**
 * @brief Divide una línea del CSV en tokens separados por comas.
 * La función maneja correctamente las comillas, permitiendo que los campos que contienen comas dentro de comillas sean tratados como un solo token. 
 * También elimina los caracteres de retorno de carro (\r) que pueden estar presentes al final de las líneas en algunos sistemas operativos.
 * 
 * @param line La línea del CSV a dividir.
 * @return Un vector de strings con los tokens resultantes.
 */
std::vector<std::string> splitCsvLineSimple(const std::string& line)
{
    std::vector<std::string> tokens;
    std::string currentToken;
    bool insideQuotes = false;

    // Recorre cada carácter de la línea, construyendo los tokens según las reglas del formato CSV
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char currentChar = line[i];

        if (currentChar == '"') {
            // Si estamos dentro de comillas y encontramos una comilla doble seguida de otra comilla doble, agregamos una comilla al token actual y avanzamos el índice para saltar la segunda comilla.
            if (insideQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                currentToken.push_back('"');
                ++i;
            } else {
                insideQuotes = !insideQuotes;
            }
            continue;
        }
        // Si encontramos una coma y no estamos dentro de comillas, significa que hemos terminado un token, así que lo agregamos al vector de tokens y limpiamos el token actual para comenzar a construir el siguiente.
        if (currentChar == ',' && !insideQuotes) {
            tokens.push_back(currentToken);
            currentToken.clear();
            continue;
        }

        // Agrega el carácter actual al token en construcción, excepto si es un retorno de carro, que se ignora para evitar problemas con líneas terminadas en \r\n.
        if (currentChar != '\r') {
            currentToken.push_back(currentChar);
        }
    }

    tokens.push_back(currentToken);
    return tokens;
}

/**
 * @brief Limpia un token que está entre comillas.
 * 
 * @param token El token a limpiar.
 * @return El token limpio.
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

/**
 * @brief Convierte un token a float, devolviendo NaN si el token está vacío o no es un número válido.
 * 
 * @param token El token a convertir.
 * @return El valor float resultante o NaN si la conversión no fue posible.
 */
float parseFloatOrNan(const std::string& token)
{
    const std::string cleanedToken = cleanQuotedToken(token);

    if (cleanedToken.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    try {
        std::size_t processedCharacters = 0;
        const float parsedValue = std::stof(cleanedToken, &processedCharacters);

        if (processedCharacters != cleanedToken.size()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        return parsedValue;
    }
    catch (...) {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

/**
 * @brief Convierte un token a entero, devolviendo -1 si el token está vacío o no es un número válido.
 * 
 * @param token El token a convertir.
 * @param parsedValue La variable donde se almacenará el valor entero resultante.
 * @return true si la conversión fue exitosa, false en caso contrario.
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

/**
 * @brief Carga un dataset desde un archivo CSV.
 * Lee el archivo línea por línea, dividiendo cada línea en campos utilizando splitCsvLineSimple.
    Para cada línea, verifica que tenga al menos el número mínimo de columnas requeridas. Si no es así, incrementa el contador de filas descartadas y continúa con la siguiente línea.
    Luego, procesa cada campo relevante (tailNum, originSeqId, originAirportCode, destSeqId, destAirportCode, depDelay, arrDelay, weatherDelay), limpiando los datos, 
    manejando los casos de valores faltantes o no válidos, y almacenando los datos en la estructura DatasetColumns. 
    También actualiza el resumen de la carga con estadísticas sobre los datos procesados, como el número de filas leídas, almacenadas, descartadas, 
    y los casos de datos faltantes para cada campo. Al final, calcula el número de IDs de secuencia únicos para origen y destino, y verifica que se hayan almacenado filas válidas antes de devolver el resultado de la carga.
 * 
 * @param filename El nombre del archivo CSV.
 * @param dataset El dataset donde se almacenarán los datos.
 * @param summary El resumen de la carga.
 * @param errorMessage La cadena de error en caso de fallo.
 * @return true si la carga fue exitosa, false en caso contrario.
 */
bool loadDataset(
    const std::string& filename,
    DatasetColumns& dataset,
    LoadSummary& summary,
    std::string& errorMessage)
{
    clearDataset(dataset);
    summary = LoadSummary{};
    errorMessage.clear();

    std::ifstream file(filename.c_str());

    if (!file.is_open()) {
        errorMessage = "No se pudo abrir el archivo CSV indicado.";
        return false;
    }

    std::string headerLine;

    if (!std::getline(file, headerLine)) {
        errorMessage = "El archivo CSV esta vacio o no se pudo leer la cabecera.";
        return false;
    }

    std::unordered_set<int> originSeqIds;
    std::unordered_set<int> destSeqIds;
    std::string line;

    // Lee el archivo línea por línea, procesando cada línea para extraer los datos relevantes y actualizar el dataset y el resumen de la carga
    while (std::getline(file, line)) {
        ++summary.rowsRead;

        const std::vector<std::string> fields = splitCsvLineSimple(line); // Divide la línea en campos utilizando la función splitCsvLineSimple

        if (fields.size() < CSV_MIN_COLUMN_COUNT) {
            ++summary.discardedRows; // Si la línea no tiene el número mínimo de columnas, se descarta y se incrementa el contador de filas descartadas en el resumen
            continue;
        }

        const std::string tailNum = cleanQuotedToken(fields[IDX_TAIL_NUM]);

        if (tailNum.empty()) {
            ++summary.missingTailNum; // Si el número de cola está vacío, se incrementa el contador de valores faltantes y se continúa con la siguiente línea
        }

        dataset.tailNum.push_back(tailNum);

        int originSeqId = -1;

        if (!parseIntFromFloatToken(fields[IDX_ORIGIN_SEQ_ID], originSeqId)) {
            ++summary.missingOriginSeqId; // Si el ID de secuencia de origen no es válido, se incrementa el contador de valores faltantes y se continúa con la siguiente línea
        }

        dataset.originSeqId.push_back(originSeqId);

        const std::string originAirportCode = cleanQuotedToken(fields[IDX_ORIGIN_AIRPORT]);

        if (originAirportCode.empty()) {
            ++summary.missingOriginAirportCode; // Si el código de aeropuerto de origen está vacío, se incrementa el contador de valores faltantes y se continúa con la siguiente línea
        }

        rememberAirportCode(dataset.originIdToCode, originSeqId, originAirportCode); // Se recuerda el código de aeropuerto asociado al ID de secuencia de origen para futuras referencias

        if (originSeqId >= 0) {
            originSeqIds.insert(originSeqId);
        }

        int destSeqId = -1;

        if (!parseIntFromFloatToken(fields[IDX_DEST_SEQ_ID], destSeqId)) {
            ++summary.missingDestSeqId; // Si el ID de secuencia de destino no es válido, se incrementa el contador de valores faltantes y se continúa con la siguiente línea
        }

        dataset.destSeqId.push_back(destSeqId);

        const std::string destAirportCode = cleanQuotedToken(fields[IDX_DEST_AIRPORT]);

        if (destAirportCode.empty()) {
            ++summary.missingDestAirportCode; // Si el código de aeropuerto de destino está vacío, se incrementa el contador de valores faltantes y se continúa con la siguiente línea
        }

        rememberAirportCode(dataset.destIdToCode, destSeqId, destAirportCode);

        if (destSeqId >= 0) {
            destSeqIds.insert(destSeqId);
        }

        // Se procesan los campos de retraso de salida, retraso de llegada y retraso por clima, convirtiendo los valores a float y manejando los casos de valores faltantes o no válidos, actualizando el resumen de la carga en consecuencia
        const float depDelay = parseFloatOrNan(fields[IDX_DEP_DELAY]);
        countMissingFloat(depDelay, summary.missingDepDelay);
        dataset.depDelay.push_back(depDelay);

        const float arrDelay = parseFloatOrNan(fields[IDX_ARR_DELAY]);
        countMissingFloat(arrDelay, summary.missingArrDelay);
        dataset.arrDelay.push_back(arrDelay);

        const float weatherDelay = parseFloatOrNan(fields[IDX_WEATHER_DELAY]);
        countMissingFloat(weatherDelay, summary.missingWeatherDelay);
        dataset.weatherDelay.push_back(weatherDelay);

        ++summary.storedRows;
    }

    summary.uniqueOriginSeqIds = originSeqIds.size(); //Tamaño del conjunto de IDs de secuencia de origen únicos encontrados durante la carga
    summary.uniqueDestinationSeqIds = destSeqIds.size(); //Tamaño del conjunto de IDs de secuencia de destino únicos encontrados durante la carga

    // Si todo falla (espermos que no :) y no se pudo almacenar ninguna fila válida, se devuelve un mensaje de error y se limpia el dataset para evitar que queden datos inconsistentes)
    if (summary.storedRows == 0) {
        errorMessage = "El CSV se ha leido, pero no se ha podido almacenar ninguna fila valida.";
        clearDataset(dataset);
        return false;
    }

    return true;
}
