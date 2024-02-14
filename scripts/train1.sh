#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=output/slurm/train1.out
#SBATCH --error=output/slurm/train1.err
#SBATCH --nodelist=slurmnode2
source activate ot

# Your job commands go here
python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml 
python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml 
python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yam