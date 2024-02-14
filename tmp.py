paths = ['data/ucf101/annotations/test_few_shot.txt',
        'data/ucf101/annotations/train_few_shot.txt',
        'data/ucf101/annotations/val_few_shot.txt']
for path in paths:
    with open(path, 'r') as f:
        data = f.readlines()
    data = [row.strip() for row in data]
    data = [row.replace("videos/", "") for row in data]
    import pdb; pdb.set_trace()
    with open(path, 'w') as f:
        data = '\n'.join(data)
        f.write(data)
    