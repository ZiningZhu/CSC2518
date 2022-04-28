#!/bin/bash 
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --output=slurm/slurm_%j.out

#SBATCH --dependency=afterok:6953219

module load transformers4

#python main.py --setting baseline
python main.py --setting uniform_noise_per_utterance --seed 42 --uniform_noise std 0.01
#python main.py --setting adaptive_noise_per_utterance --seed $1 --scaling_factor 0.3
#python main.py --setting multimodal_noise_per_utterance --seed $1 --scaling_factor 0.3 --gmm_n_components 3