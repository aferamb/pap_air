// kernels.cuh
// Declaración de kernels GPU para la práctica

#pragma once

// Kernel de prueba inicial: imprime el ID de cada hilo
__global__ void holaGPU();

// Fase 01: retraso en despegues
__global__ void depDelayKernel(float* dep_delay, int n, float umbral);