#!/bin/bash

#SBATCH --job-name=metal_pka_transfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --account=MST111483
#SBATCH --output=logs/job_output_%j.txt
#SBATCH --error=logs/job_error_%j.txt   

python -u ../model/metal_pka_transfer.py --version pka_ver14