import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

random_seed = 10
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)


file_list = list(Path('results').rglob('*.png'))
keys = set(map(lambda x:x.name.split('_')[0], file_list))
sorted_list = [[y for y in file_list if y.name.split('_')[0]==x] for x in keys]

print(sorted_list[0])

ensemble_list = ['NAFNet']

path = 'datasets/Dacon/train/lr_ensemble_concat'
os.makedirs(path, exist_ok=True)

# weights_list = [0.3, 1.8, 0.9]

for files in tqdm(sorted_list):
    concat_list = []
    count = 0
    for f in files:
        if any(exp in str(f) for exp in ensemble_list):
            if '_gt' in f.name:
                continue
            temp = cv2.imread(str(f))
            concat_list.append(temp)
            count += 1
    if not len(concat_list):
        continue
    concat_img = np.concatenate(concat_list, axis=2)
    np.save(f"{path}/{files[0].stem}.npy", concat_img)


path = 'datasets/Dacon/train/lr_ensemble_gt'
os.makedirs(path, exist_ok=True)
file_list = list(Path('datasets/Dacon/train/lr_random').rglob('*.png'))

for file in tqdm(file_list):
    temp = cv2.imread(str(file))
    np.save(f"{path}/{file.stem}.npy", temp)
