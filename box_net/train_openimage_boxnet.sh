#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --output=oi-c.out
#SBATCH --job-name=Box_net_training
#SBATCH --mail-user=11mhg@queensu.ca
#SBATCH --mail-type=ALL


time python3 train.py --batch_size 32 --data_dir /home/mhg1/Dataset/OpenImage/tfrecords/*train* --val_dir /home/mhg1/Dataset/OpenImage/tfrecords/*val* --labels ./OpenImage.txt --model_dir ./open_dir/ --num_epochs 55 --steps_per_epoch 50000 --logs ./logs/
