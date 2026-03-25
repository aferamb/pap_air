#pragma once

#include <cstddef>
#include <string>
#include <vector>

/*
    csv_reader.h

    Este modulo concentra toda la logica de Fase 0 relacionada con:

    - abrir el CSV del dataset desde disco;
    - validar que la cabecera coincide con el formato esperado;
    - limpiar datos de texto y numericos;
    - convertir el CSV a una estructura por columnas en memoria del host;
    - devolver estadisticas utiles para depuracion, pruebas y defensa.

    El diseno por columnas se ha elegido porque la practica pide preparar
    vectores simples para GPU y este formato hace mas directa esa conversion.
*/

/*
    DatasetColumns

    Estructura principal en memoria del host. Cada vector representa una
    columna del dataset y todas las posiciones deben permanecer alineadas:

    - el indice i de cualquier vector representa siempre la misma fila logica;
    - si un dato numerico falta, se guarda como NAN;
    - si un dato de texto falta, se guarda como cadena vacia;
    - si un identificador entero falta, se usa -1 como centinela sencillo.

    Este contrato es importante porque las fases posteriores prepararan buffers
    para GPU asumiendo que todas las columnas comparten el mismo indice base.
*/
struct DatasetColumns {
    // Retraso de salida del vuelo en minutos.
    std::vector<float> depDelay;

    // Retraso real de llegada en minutos.
    std::vector<float> arrDelay;

    // Retraso atribuido al tiempo meteorologico.
    std::vector<float> weatherDelay;

    // Marca temporal de despegue.
    std::vector<float> depTime;

    // Marca temporal de aterrizaje.
    std::vector<float> arrTime;

    // Matricula del avion asociada a cada fila.
    std::vector<std::string> tailNum;

    // Identificador numerico del aeropuerto de origen.
    std::vector<int> originSeqId;

    // Identificador numerico del aeropuerto de destino.
    std::vector<int> destSeqId;

    // Codigo textual del aeropuerto de origen.
    std::vector<std::string> originAirport;

    // Codigo textual del aeropuerto de destino.
    std::vector<std::string> destAirport;
};

/*
    CsvLoadStats

    Resume que ha ocurrido durante la carga del fichero. Estas cifras sirven
    para validar que el lector esta limpiando y almacenando datos como se espera.
*/
struct CsvLoadStats {
    // Numero total de filas de datos leidas tras la cabecera.
    std::size_t dataRowsRead = 0;

    // Numero total de filas que finalmente se han almacenado.
    std::size_t storedRows = 0;

    // Numero de filas descartadas por problemas estructurales.
    std::size_t discardedRows = 0;

    // Numero de filas con menos columnas de las esperadas.
    std::size_t shortRows = 0;

    // Contadores de faltantes por columna. Son utiles para saber la calidad
    // real del dataset cargado y justificar decisiones de limpieza.
    std::size_t missingTailNum = 0;
    std::size_t missingOriginSeqId = 0;
    std::size_t missingOriginAirport = 0;
    std::size_t missingDestSeqId = 0;
    std::size_t missingDestAirport = 0;
    std::size_t missingDepTime = 0;
    std::size_t missingDepDelay = 0;
    std::size_t missingArrTime = 0;
    std::size_t missingArrDelay = 0;
    std::size_t missingWeatherDelay = 0;

    // Numero de SEQ_ID unicos detectados para origen y destino. Este sera el
    // criterio principal del estado inicial porque es el mismo que se usa en
    // la Fase 04 para construir el histograma en GPU.
    std::size_t uniqueOriginAirportSeqIds = 0;
    std::size_t uniqueDestinationAirportSeqIds = 0;

    // Numero de codigos textuales unicos detectados para origen y destino.
    // Se conserva porque ayuda a explicar por que el dataset muestra 375
    // codigos pero 409 SEQ_ID distintos.
    std::size_t uniqueOriginAirportCodes = 0;
    std::size_t uniqueDestinationAirportCodes = 0;
};

/*
    CsvLoadResult

    Objeto de salida de la Fase 0. Agrupa:

    - resultado correcto o incorrecto de la carga;
    - ruta del fichero procesado;
    - mensaje de error explicativo si algo falla;
    - cabecera validada;
    - dataset ya limpio;
    - estadisticas de carga.

    Esta estructura evita devolver muchos valores sueltos y simplifica el uso
    desde main.cu y desde la pantalla de estado de la aplicacion.
*/
struct CsvLoadResult {
    bool success = false;
    std::string filePath;
    std::string errorMessage;
    std::vector<std::string> header;
    DatasetColumns dataset;
    CsvLoadStats stats;
};

/*
    splitCsvLineSimple

    Divide una linea CSV en tokens con un parser sencillo, suficiente para el
    dataset actual. Soporta:

    - campos vacios;
    - comillas simples alrededor de un campo;
    - comillas dobles escapadas dentro de un campo.

    No pretende cubrir todo el estandar CSV, solo el caso practico de este
    dataset sin recurrir a librerias externas.
*/
std::vector<std::string> splitCsvLineSimple(const std::string& line);

/*
    cleanQuotedToken

    Limpia un token textual eliminando espacios exteriores y, si existen,
    comillas envolventes basicas. Esta funcion se usa antes de validar cabeceras
    y antes de interpretar cualquier campo del CSV.
*/
std::string cleanQuotedToken(const std::string& token);

/*
    parseFloatOrNan

    Intenta interpretar un token como float. Si el campo esta vacio o no puede
    convertirse de forma segura, devuelve NAN para preservar la ausencia real
    del dato y no introducir valores inventados.
*/
float parseFloatOrNan(const std::string& token);

/*
    parseIntFromFloatToken

    Convierte un token del CSV con formato decimal a entero truncado. Devuelve
    false si el campo falta o es invalido. En ese caso deja parsedValue en -1
    como centinela simple para campos como ORIGIN_SEQ_ID y DEST_SEQ_ID.
*/
bool parseIntFromFloatToken(const std::string& token, int& parsedValue);

/*
    validateHeader

    Comprueba que la cabecera leida contiene, en las posiciones esperadas, las
    columnas que necesita la practica. El objetivo es detectar cuanto antes
    cambios de formato o ficheros incorrectos.
*/
bool validateHeader(const std::vector<std::string>& header);

/*
    loadDataset

    Funcion principal de Fase 0. Abre el fichero indicado, valida cabecera,
    limpia fila a fila, actualiza estadisticas y devuelve el dataset completo
    listo para ser reutilizado por el resto de la aplicacion.
*/
CsvLoadResult loadDataset(const std::string& filename);

/*
    getDatasetRowCount

    Devuelve el numero de filas almacenadas. Se usa como tamano base del dataset
    para resumenes de estado y para preparar configuraciones de lanzamiento
    futuras en GPU.
*/
std::size_t getDatasetRowCount(const DatasetColumns& dataset);
