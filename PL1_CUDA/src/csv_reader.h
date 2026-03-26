#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

/*
    csv_reader.h

    Este modulo se ha simplificado para quedarse solo con lo que hoy usa la
    practica:

    - las columnas reales de las Fases 01, 02, 03 y 04;
    - un resumen pequeno de la carga.

    La cabecera ya no intenta describir un lector generico. El CSV de la
    practica es fijo y conocido, asi que la carga se apoya en indices de
    columna constantes y en una limpieza basica en una sola pasada.
*/

/*
    DatasetColumns

    Estructura principal del dataset en memoria del host. Se guardan solo las
    columnas que el resto del proyecto necesita de verdad:

    - retrasos para Fases 01, 02 y 03;
    - matricula para Fase 02;
    - IDs de aeropuerto para Fase 04;
    - mapas ID -> codigo para poder imprimir el histograma final en CPU.

    Los vectores siguen alineados por indice: la posicion i representa siempre
    la misma fila logica del CSV.
*/
struct DatasetColumns {
    std::vector<float> depDelay;
    std::vector<float> arrDelay;
    std::vector<float> weatherDelay;
    std::vector<std::string> tailNum;
    std::vector<int> originSeqId;
    std::vector<int> destSeqId;
    std::unordered_map<int, std::string> originIdToCode;
    std::unordered_map<int, std::string> destIdToCode;
};

/*
    LoadSummary

    Resumen compacto de la Fase 0. Conserva solo las cifras que siguen siendo
    utiles en el flujo actual:

    - cuantas filas se leyeron y cuantas quedaron almacenadas;
    - cuantas se descartaron por estructura corta;
    - cuantos faltantes hay en todas las columnas que hoy se usan en las
      fases posteriores;
    - cuantos aeropuertos unicos aparecen por SEQ_ID en origen y destino.
*/
struct LoadSummary {
    std::size_t rowsRead = 0;
    std::size_t storedRows = 0;
    std::size_t discardedRows = 0;
    std::size_t missingTailNum = 0;
    std::size_t missingOriginSeqId = 0;
    std::size_t missingOriginAirportCode = 0;
    std::size_t missingDestSeqId = 0;
    std::size_t missingDestAirportCode = 0;
    std::size_t missingDepDelay = 0;
    std::size_t missingArrDelay = 0;
    std::size_t missingWeatherDelay = 0;
    std::size_t uniqueOriginSeqIds = 0;
    std::size_t uniqueDestinationSeqIds = 0;
};

/*
    loadDataset

    Carga el CSV, limpia los campos necesarios y deja el dataset y su resumen
    listos para el resto del programa. Devuelve true si la carga produjo al
    menos una fila valida.
*/
bool loadDataset(
    const std::string& filename,
    DatasetColumns& dataset,
    LoadSummary& summary,
    std::string& errorMessage);
