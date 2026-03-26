#include "csv_reader.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>

/*
    csv_reader.cpp

    La carga se hace en una sola pasada sobre el fichero:

    1. se abre el CSV;
    2. se descarta la cabecera;
    3. se leen solo las columnas que usa hoy la practica;
    4. los faltantes numericos se guardan como NAN;
    5. los IDs ausentes se guardan como -1;
    6. los codigos de aeropuerto se conservan solo en mapas ID -> codigo.

    La idea de esta version es ser mas corta y mas directa. El CSV del proyecto
    es fijo, asi que no hace falta una validacion pesada de cabecera.
*/

namespace {

constexpr std::size_t CSV_MIN_COLUMN_COUNT = 14;
constexpr std::size_t IDX_TAIL_NUM = 3;
constexpr std::size_t IDX_ORIGIN_SEQ_ID = 5;
constexpr std::size_t IDX_ORIGIN_AIRPORT = 6;
constexpr std::size_t IDX_DEST_SEQ_ID = 7;
constexpr std::size_t IDX_DEST_AIRPORT = 8;
constexpr std::size_t IDX_DEP_DELAY = 10;
constexpr std::size_t IDX_ARR_DELAY = 12;
constexpr std::size_t IDX_WEATHER_DELAY = 13;

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

void countMissingFloat(float value, std::size_t& counter)
{
    if (std::isnan(value)) {
        ++counter;
    }
}

void rememberAirportCode(std::unordered_map<int, std::string>& idToCode, int seqId, const std::string& code)
{
    if (seqId >= 0 && !code.empty() && idToCode.find(seqId) == idToCode.end()) {
        idToCode[seqId] = code;
    }
}

} // namespace

std::vector<std::string> splitCsvLineSimple(const std::string& line)
{
    std::vector<std::string> tokens;
    std::string currentToken;
    bool insideQuotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char currentChar = line[i];

        if (currentChar == '"') {
            if (insideQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                currentToken.push_back('"');
                ++i;
            } else {
                insideQuotes = !insideQuotes;
            }
            continue;
        }

        if (currentChar == ',' && !insideQuotes) {
            tokens.push_back(currentToken);
            currentToken.clear();
            continue;
        }

        if (currentChar != '\r') {
            currentToken.push_back(currentChar);
        }
    }

    tokens.push_back(currentToken);
    return tokens;
}

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

    while (std::getline(file, line)) {
        ++summary.rowsRead;

        const std::vector<std::string> fields = splitCsvLineSimple(line);

        if (fields.size() < CSV_MIN_COLUMN_COUNT) {
            ++summary.discardedRows;
            continue;
        }

        const std::string tailNum = cleanQuotedToken(fields[IDX_TAIL_NUM]);

        if (tailNum.empty()) {
            ++summary.missingTailNum;
        }

        dataset.tailNum.push_back(tailNum);

        int originSeqId = -1;

        if (!parseIntFromFloatToken(fields[IDX_ORIGIN_SEQ_ID], originSeqId)) {
            ++summary.missingOriginSeqId;
        }

        dataset.originSeqId.push_back(originSeqId);

        const std::string originAirportCode = cleanQuotedToken(fields[IDX_ORIGIN_AIRPORT]);

        if (originAirportCode.empty()) {
            ++summary.missingOriginAirportCode;
        }

        rememberAirportCode(dataset.originIdToCode, originSeqId, originAirportCode);

        if (originSeqId >= 0) {
            originSeqIds.insert(originSeqId);
        }

        int destSeqId = -1;

        if (!parseIntFromFloatToken(fields[IDX_DEST_SEQ_ID], destSeqId)) {
            ++summary.missingDestSeqId;
        }

        dataset.destSeqId.push_back(destSeqId);

        const std::string destAirportCode = cleanQuotedToken(fields[IDX_DEST_AIRPORT]);

        if (destAirportCode.empty()) {
            ++summary.missingDestAirportCode;
        }

        rememberAirportCode(dataset.destIdToCode, destSeqId, destAirportCode);

        if (destSeqId >= 0) {
            destSeqIds.insert(destSeqId);
        }

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

    summary.uniqueOriginSeqIds = originSeqIds.size();
    summary.uniqueDestinationSeqIds = destSeqIds.size();

    if (summary.storedRows == 0) {
        errorMessage = "El CSV se ha leido, pero no se ha podido almacenar ninguna fila valida.";
        clearDataset(dataset);
        return false;
    }

    return true;
}

std::size_t getDatasetRowCount(const DatasetColumns& dataset)
{
    return dataset.depDelay.size();
}
