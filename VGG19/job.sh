#!/bin/bash
#SBATCH --job-name=VGG19                   		 # Job name
#SBATCH --output=%j_output.txt                   # Standard output file
#SBATCH --error=%j_error.txt                     # Standard error file
#SBATCH --ntasks=1              		         # Number of tasks (usually = 1 for GPU jobs)
#SBATCH --cpus-per-task=8       		         # Number of CPU cores per task
#SBATCH --gres=gpu:1            		         # Request 1 GPU
#SBATCH --mem=48G               		         # Memory requested (48GB in this case)
#SBATCH --time=7-00:00:00       		         # Maximum time requested (7 days)
#SBATCH --partition=long                         # Partition name


python vgg.py
