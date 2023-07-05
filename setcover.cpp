#include <iostream>
#include <vector>
#include <omp.h>
#include "include/BasicCDS.h"

using namespace std;
using namespace cds;

// Función para calcular el "set cover problem" de manera paralela
vector<int> calculateSetCoverProblemParallel(ulong* sets, int numSets, int numElements) {
    vector<int> selectedSets;
    ulong coveredElements = 0;
    bool allElementsCovered = false;

    // Mientras no se hayan cubierto todos los elementos
    while (!allElementsCovered) {
        int bestSet = -1;
        int maxCoveredElements = 0;

        // Encontrar el conjunto que cubra la mayor cantidad de elementos no cubiertos
        #pragma omp parallel for
        for (int i = 0; i < numSets; ++i) {
            if ((coveredElements & sets[i]) != sets[i]) {
                int covered = __builtin_popcountl(sets[i] & ~coveredElements);
                #pragma omp critical
                {
                    if (covered > maxCoveredElements) {
                        maxCoveredElements = covered;
                        bestSet = i;
                    }
                }
            }
        }

        // Si se encontró un conjunto que cubre elementos no cubiertos
        if (bestSet != -1) {
            #pragma omp critical
            {
                selectedSets.push_back(bestSet);
                coveredElements |= sets[bestSet];
            }
        } else {
            // Si no se encontró un conjunto que cubra más elementos, se han cubierto todos los elementos
            allElementsCovered = true;
        }
    }

    return selectedSets;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Uso: " << argv[0] << " n m" << endl;
        return 1;
    }

    int numSets = atoi(argv[1]);
    int numElements = atoi(argv[2]);

    // Crear y llenar los conjuntos con bits aleatorios (para ejemplo)
    ulong* sets = new ulong[numSets];
    for (int i = 0; i < numSets; ++i) {
        for (int j = 0; j < numElements; ++j) {
            if (rand() % 2 == 1) {
                setBit64(&sets[i], j);
                //setBit64(sets, i * numElements + j);
            }
        }
        printBitsUlong(sets[i]);
        cout << endl;
    }


    // Medir el tiempo de ejecución
    double startTime = omp_get_wtime();

    // Calcular el "set cover problem" de manera paralela
    vector<int> selectedSets = calculateSetCoverProblemParallel(sets, numSets, numElements);

    // Calcular el tiempo transcurrido
    double endTime = omp_get_wtime();
    double executionTime = endTime - startTime;

    // Imprimir los conjuntos que forman la cobertura completa
    cout << "Conjuntos que forman la cobertura completa:" << endl;
    for (int i : selectedSets) {
        cout << "Conjunto " << i << ": ";
        printBitsUlong(sets[i]);
        cout << endl;
    }

    // Imprimir el tiempo de ejecución
    cout << "Tiempo de ejecución: " << executionTime << " segundos" << endl;

    // Liberar memoria
    delete[] sets;

    return 0;
}

//g++ -std=c++14 -O3 -Wall -DVERBOSE -I./include/ ./include/BasicCDS.cpp setcover.cpp -o setcover -fopenmp
