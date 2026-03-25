#pragma once

/*
    kernels.cuh

    Este header declara el kernel CUDA mas simple que existe ahora mismo en el
    repositorio. Aunque la Fase 0 no lo ejecuta desde main, el kernel se sigue
    conservando porque representa la base actual de una reduccion elemental.
*/

/*
    reductionSimple

    Kernel de reduccion simple asociado a la variante 3.1 de la Fase 03.

    Parametros:

    - data: vector de entrada en memoria global de GPU;
    - result: puntero a un unico acumulador global en device;
    - n: numero total de elementos del vector;
    - isMax: selector de operacion, true para maximo y false para minimo.

    Cada hilo:

    1. calcula su indice global 1D;
    2. comprueba si ese indice esta dentro del rango;
    3. lee su elemento de entrada;
    4. aplica una operacion atomica sobre el acumulador global.

    Es un enfoque didactico y funcional, pero no el mas eficiente, porque todos
    los hilos compiten por la misma posicion de memoria global.
*/
__global__ void reductionSimple(int* data, int* result, int n, bool isMax);
