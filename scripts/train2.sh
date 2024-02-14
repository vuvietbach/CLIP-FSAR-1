#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=output/slurm/train2.out
#SBATCH --error=output/slurm/train2.err
#SBATCH --nodelist=slurmnode3
source activate ot

# Your job commands go here
python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml TEMPO_PRIOR 2.0 
python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml TEMPO_PRIOR 2.0 
python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml
