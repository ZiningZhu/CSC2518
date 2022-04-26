#!/bin/bash 
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --output=slurm_%j.out

module load transformers4

#python main.py --setting baseline
#python main.py --setting uniform_noise_per_utterance
#python main.py --setting adaptive_noise_per_utterance
python main.py --setting multimodal_noise_per_utterance