#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <cmath>
#include "include/BasicCDS.h"

using namespace std;
using namespace cds;

int main() {
    ulong *A = new ulong[100];

    // Initialize the array
    for (ulong i = 0; i < 10; i++) {
        cleanBit64(A, i);
    }

    // Change an element of the array using setBit64
    ulong indiceBit = 0;  // Index of the bit to change (0-63)
    setBit64(A, indiceBit);

    // Print the modified array
    printBitsUlong(*A);

    cout << endl << (getNum64(A, indiceBit, 1)==1)<< endl;

    delete[] A;  // Deallocate the memory

    return 0;
}
