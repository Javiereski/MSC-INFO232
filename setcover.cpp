#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <cmath>
#include <cuda.h>
#include "include/BasicCDS.h"

using namespace std;
using namespace cds;

// Función de kernel para calcular el Set Cover paralelizado
__global__ void setCoverKernel(ulong *conjunto, const int* elementos, int* resultado, int numConjuntos, int numElementos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElementos) {
        for (int i = 0; i < numConjuntos; i++) {
            if (getNum64(C, i * numElementos + idx, 1) == 1){
                resultado[idx] = i;
                break;
            }
        }
    }
}

// Función principal
int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Uso: programa <numero_de_conjuntos> <numero_de_elementos>" << endl;
        return 1;
    }

    // Obtener el número de conjuntos y elementos de los argumentos de línea de comandos
    int numConjuntos = stoi(argv[1]);
    int numElementos = stoi(argv[2]);

    // Generar la matriz de conjuntos aleatoriamente
    //vector<int> conjuntos(numConjuntos * numElementos);
    srand(time(nullptr));  // Inicializar la semilla del generador de números aleatorios

    ulong *C = new ulong[numConjuntos * numElementos]:

    for (int i = 0; i < numConjuntos; i++) {
        for (int j = 0; j < numElementos; j++) {
            if rand() % 2{
                setBit64(&C, i * numElementos + j):
            }else{
                cleanBit64(&C, i * numElementos + j):
            }
            //conjuntos[i * numElementos + j] = rand() % 2;  // Generar 0 o 1 aleatoriamente
        }
    }

    // Definir los elementos
    vector<int> elementos(numElementos);
    for (int i = 0; i < numElementos; i++) {
        elementos[i] = i + 1;
    }

    // Calcular el tamaño en bytes de los arreglos
    size_t sizeConjuntos = numConjuntos * numElementos * sizeof(int);
    size_t sizeElementos = numElementos * sizeof(int);
    size_t sizeResultado = numElementos * sizeof(int);

    // Reservar memoria en el dispositivo (GPU)
    int* d_conjuntos;
    int* d_elementos;
    int* d_resultado;
    cudaMalloc((void**)&d_conjuntos, sizeConjuntos);
    cudaMalloc((void**)&d_elementos, sizeElementos);
    cudaMalloc((void**)&d_resultado, sizeResultado);

    // Copiar los datos desde la CPU al dispositivo
    cudaMemcpy(d_conjuntos, conjuntos.data(), sizeConjuntos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elementos, elementos.data(), sizeElementos, cudaMemcpyHostToDevice);

    // Definir la configuración de ejecución del kernel
    int threadsPerBlock = 256;
    int numBlocks = ceil((float)numElementos / threadsPerBlock);

    // Lanzar el kernel en la GPU
    setCoverKernel<<<numBlocks, threadsPerBlock>>>(d_conjuntos, d_elementos, d_resultado, numConjuntos, numElementos);

    // Copiar el resultado desde el dispositivo a la CPU
    vector<int> resultado(numElementos);
    cudaMemcpy(resultado.data(), d_resultado, sizeResultado, cudaMemcpyDeviceToHost);

    // Imprimir el resultado
    cout << "Resultado del Set Cover:" << endl;
    for (int i = 0; i < numElementos; i++) {
        cout << "Elemento " << elementos[i] << " está en el conjunto " << resultado[i] << endl;
    }

    // Liberar memoria en el dispositivo
    cudaFree(d_conjuntos);
    cudaFree(d_elementos);
    cudaFree(d_resultado);

    return 0;
}
