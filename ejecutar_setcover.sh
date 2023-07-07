#!/bin/bash
inicio=1
fin=12
salto=1

for i in $(seq $inicio $salto $fin); do
   ./setcover 8 $((2**$i)) 2
done
