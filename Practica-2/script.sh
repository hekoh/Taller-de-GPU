#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --partition=GPUS #cambiar a clusterX despues
#SBATCH -o output.txt
#SBATCH -e errors.xtx
#SBATCH --time=00:01:00
./EJ1

