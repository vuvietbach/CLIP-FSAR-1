python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml
python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml TEMPO_PRIOR 2.0 

python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml 
python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml TEMPO_PRIOR 2.0 

python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml 
python run.py --cfg configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml TEMPO_PRIOR 2.0

python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_5shot_v1.yaml
srun --nodelist=slurmnode7 --output=test.out --error=test.err python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_5shot_v1.yaml TEMPO_PRIOR 2.0 &