#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// Definición de las funciones para manipular los bits
extern "C" {
    void setBit64(unsigned long* array, int index);
    void cleanBit64(unsigned long* array, int index);
    int readBit64(const unsigned long array, int index);
    void printBitsUlong(const unsigned long array);
}

__global__ void setCoverProblemGPU(const unsigned long* sets, unsigned long* result,
                                   int numSets, int numElements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Si el hilo está dentro de los límites del arreglo de conjuntos
    if (tid < numSets) {
        unsigned long coveredElements = 0;

        // Comprobar si el conjunto actual cubre todos los elementos
        for (int j = 0; j < numElements; ++j) {
            if (readBit64(sets[tid], j)) {
                coveredElements |= (1UL << j);
            }
        }

        // Almacenar el resultado del conjunto en el arreglo de resultados
        result[tid] = coveredElements;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Uso: " << argv[0] << " n m" << std::endl;
        return 1;
    }

    int numSets = std::atoi(argv[1]);
    int numElements = std::atoi(argv[2]);

    // Crear y asignar memoria en la GPU para los arreglos
    unsigned long* sets = new unsigned long[numSets];
    unsigned long* result = new unsigned long[numSets];
    unsigned long* devSets, *devResult;
    cudaMalloc((void**)&devSets, numSets * sizeof(unsigned long));
    cudaMalloc((void**)&devResult, numSets * sizeof(unsigned long));

    // Llenar los conjuntos con bits aleatorios (para ejemplo)
    for (int i = 0; i < numSets; ++i) {
        for (int j = 0; j < numElements; ++j) {
            if (rand() % 2 == 1) {
                setBit64(&sets[i], j);
            }
        }
    }

    // Copiar los arreglos desde la CPU a la GPU
    cudaMemcpy(devSets, sets, numSets * sizeof(unsigned long), cudaMemcpyHostToDevice);

    // Configurar el tamaño de la rejilla y los bloques de hilos
    int blockSize = 256;
    int gridSize = (numSets + blockSize - 1) / blockSize;

    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Registrar el inicio del tiempo
    cudaEventRecord(start);

    // Ejecutar el kernel en la GPU
    setCoverProblemGPU<<<gridSize, blockSize>>>(devSets, devResult, numSets, numElements);

    // Copiar el resultado desde la GPU a la CPU
    cudaMemcpy(result, devResult, numSets * sizeof(unsigned long), cudaMemcpyDeviceToHost);

    // Registrar el fin del tiempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular el tiempo transcurrido en milisegundos
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Imprimir el tiempo transcurrido
    std::cout << "Tiempo de ejecución: " << milliseconds << " ms" << std::endl;

    // Imprimir los conjuntos que forman la cobertura completa
    std::vector<int> selectedSets;
    for (int i = 0; i < numSets; ++i) {
        if (result[i] == ((1UL << numElements) - 1)) {
            selectedSets.push_back(i);
        }
    }
    std::cout << "Conjuntos que forman la cobertura completa:" << std::endl;
    for (int i : selectedSets) {
        std::cout << "Conjunto " << i << ": ";
        printBitsUlong(sets[i]);
        std::cout << std::endl;
    }

    // Liberar memoria
    delete[] sets;
    delete[] result;
    cudaFree(devSets);
    cudaFree(devResult);

    // Destruir los eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// g++ -std=c++14 -O3 -Wall -DVERBOSE -I./include/ ./include/BasicCDS.cpp setcover.cpp -o setcover -lcudart