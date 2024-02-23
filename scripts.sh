python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml debug True
srun --output=output/slurm/ssv2/train_1shot.out --error=output/slurm/ssv2/train_1shot.err --nodelist=slurmnode6 python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml &
# debug
python run.py --cfg configs/projects/CLIPFSAR/ssv2_full/debug.yaml

find hmdb/ -name '*.rar' -exec unrar x -r {} \; 

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws s3 cp /vinserver_user/bach.vv200061/fsar/HyRSMPlusPlus/ssv2.zip s3://aiotlab-satellite