srun --output=output/slurm/train1.out --error=output/slurm/train1.err --nodelist=slurmnode2 -c "python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml ; python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml ; python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml" &
srun --output=output/slurm/train2.out --error=output/slurm/train2.err --nodelist=slurmnode3 -c "python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml TEMPO_PRIOR 2.0 || python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml TEMPO_PRIOR 2.0 || python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml" &
srun --output=output/slurm/train3.out --error=output/slurm/train3.err --nodelist=slurmnode7 -c "python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml TEMPO_PRIOR 2.0 || python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml TEMPO_PRIOR 2.0" &

python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml 
python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml debug True
python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml debug True
python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml TEMPO_PRIOR 2.0 debug True
python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml TEMPO_PRIOR 2.0 
python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml debug True
python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml TEMPO_PRIOR 2.0
python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml TEMPO_PRIOR 2.0
