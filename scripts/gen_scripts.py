def main():
    template = "python run.py --cfg {}"
    datasets = ['ssv2', 'kinetics', 'ucf101', 'hmdb51']
    datasets = [
        'configs/projects/CLIPFSAR/hmdb51/CLIPFSAR_HMDB51_1shot_v1.yaml',
        'configs/projects/CLIPFSAR/kinetics100/CLIPFSAR_K100_1shot_v1.yaml',
        'configs/projects/CLIPFSAR/ssv2_full/CLIPFSAR_SSv2_Full_1shot_v1.yaml',
        'configs/projects/CLIPFSAR/ucf101/CLIPFSAR_UCF101_1shot_v1.yaml'
    ]
    cmds = []
    video_formats = ['', '', '', '.avi']
    for i, dataset in enumerate(datasets):
        cmd = template.format(dataset)
        # sdtw
        cmds.append(cmd)
        # tpm
        cmd += " TEMPO_PRIOR 2.0"
        cmds.append(cmd)
            
    with open('scripts/1shot.sh', 'w') as f:
        f.write('\n'.join(cmds))        
        
if __name__ == '__main__':
    main()