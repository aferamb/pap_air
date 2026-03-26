#pragma once

#include <string>

/*
    dataset_gpu.cuh

    Este modulo agrupa la Fase 0 real del programa:

    - carga del CSV;
    - resumen de datos leidos;
    - preparacion de las estructuras persistentes en GPU.
*/

void liberarGPU();
void printLoadSummary();
bool cargarDataset(const std::string& datasetPath);
bool datasetListoParaGPU();
