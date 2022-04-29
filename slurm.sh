#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH -A plgimba3-cpu

module add python/3.9.6-gcccore-11.2.0

python3 -W ignore ${1} ${@:2}
