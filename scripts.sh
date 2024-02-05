srun --python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml
srun --output=output/slurm/ssv2/train_1shot.out --error=output/slurm/ssv2/train_1shot.err --nodelist=slurmnode6 python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml &
# debug
python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/debug.yaml