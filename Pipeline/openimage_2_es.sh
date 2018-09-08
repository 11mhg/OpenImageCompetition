#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=3-0
#SBATCH --output=es.out


../scratch/elasticsearch-6.3.2/bin/elasticsearch & 

sleep 5m

python -c "from test_data2 import *;ppd=PreProcessData('./OpenImage.txt');ppd.get_open_images('../Dataset/OpenImage/',data_type='train')"
