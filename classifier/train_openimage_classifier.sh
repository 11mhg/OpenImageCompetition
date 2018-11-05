#!/bin/bash
#SBATCH --job-name=Classifier_Training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --output=oi-c.out
#SBATCH --mail-user=11mhg@queensu.ca
#SBATCH --mail-type=ALL

time python3 train.py --batch_size 128 --data_dir /home/mhg1/Dataset/OpenImage/tfrecords/*train* --val_dir /home/mhg1/Dataset/OpenImage/tfrecords/*val* --labels ./OpenImage.txt --model_dir ./open_dir/ --num_epochs 75 --steps_per_epoch 100000 --logs ./logs/
