#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=output/slurm/train3.out
#SBATCH --error=output/slurm/train3.err
#SBATCH --nodelist=slurmnode7
source activate ot

python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml TEMPO_PRIOR 2.0 
python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml TEMPO_PRIOR 2.0
