#pragma once

/*
    parte4.cuh
    Declaración de la función de la Fase 04, que se encarga de procesar los aeropuertos de origen o destino, 
    dependiendo de la opción seleccionada por el usuario, y generar un histograma de conteo para cada aeropuerto utilizando memoria compartida en la GPU. 
    La función también verifica que el histograma pueda caber en la memoria compartida antes de ejecutar los kernels, 
    y maneja la reserva y liberación de memoria en la GPU para los histogramas parciales y final. 
    Finalmente, imprime el histograma resultante utilizando una función de impresión específica para esta fase.
*/

void phase04(int airportOption, int threshold);
