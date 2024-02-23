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

srun --output=output/slurm/train6.out --error=output/slurm/train6.err --nodelist=slurmnode6 python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml &
srun --output=output/slurm/train2.out --error=output/slurm/train2.err --nodelist=slurmnode2 python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml TEMPO_PRIOR 2.0 &
srun --output=output/slurm/train7.out --error=output/slurm/train7.err --nodelist=slurmnode7 python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml &
srun --output=output/slurm/train4.out --error=output/slurm/train4.err --nodelist=slurmnode4 python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml TEMPO_PRIOR 2.0 &


python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml test_only True checkpoint_loadpath output/hmdb51/sdtw/1shot/14-02_21-37/it200_acc82.6500015258789.pt && \
python run.py --cfg configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml TEMPO_PRIOR 2.0 test_only True checkpoint_loadpath output/hmdb51/tpm2.0/1shot/14-02_21-37/it200_acc81.3499984741211.pt && \
python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml TEMPO_PRIOR 2.0 test_only True checkpoint_loadpath output/ssv2/tpm2.0/1shot/14-02_21-37/it600_acc49.04999923706055.pt

python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml test_only True checkpoint_loadpath output/kinetics/sdtw/1shot/15-02_22-17/it600_acc89.3499984741211.pt && \
python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml TEMPO_PRIOR 2.0 test_only True checkpoint_loadpath output/kinetics/tpm2.0/1shot/15-02_22-17/it600_acc89.9000015258789.pt

python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml test_only True checkpoint_loadpath output/ssv2/sdtw/1shot/15-02_22-17/it600_acc53.70000076293945.pt

python run.py --cfg configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml debug True