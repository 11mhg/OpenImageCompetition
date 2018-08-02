#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --output=oi.out


python3 ./make_openimage.py
