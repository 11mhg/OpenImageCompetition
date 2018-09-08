#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=1-0
#SBATCH --output=indexing.out
#SBATCH --mail-user=11mhg@queensu.ca
#SBATCH --mail-type=END


~/scratch/elasticsearch-6.3.2/bin/elasticsearch &

sleep(120)


python -c "from data.data import *;ppd=PreProcessData('./OpenImage.txt');ppd.get_open_images('../Dataset/OpenImage/',data_type='val');from image_index import *;convert_to(ppd,'../Dataset/OpenImage/','OpenImage',data_type='val')"
