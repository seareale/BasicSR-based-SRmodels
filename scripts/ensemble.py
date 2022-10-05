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

if Path('results/ensemble').exists():
    shutil.rmtree(Path('results/ensemble'))

file_list = list(Path('results').rglob('*.png'))
keys = set(map(lambda x:x.name.split('_')[0], file_list))
sorted_list = [[y for y in file_list if y.name.split('_')[0]==x] for x in keys]

ensemble_list = [str(x) for x in Path('results').iterdir() if not x.is_file() and 'test_NAFNet' in str(x)]
print(ensemble_list)

path = 'results/ensemble'
os.makedirs(path, exist_ok=True)

# weights_list = [0.3, 1.8, 0.9]

for files in tqdm(sorted_list):
    img = np.zeros((512,512,3))
    count = 0
    # weights_list = [1] * len(files)
    for f in files:
        if any(exp in str(f) for exp in ensemble_list):
            if '_gt' in f.name:
                continue
            temp = cv2.imread(str(f)).astype(np.float32)
            img += temp # * weights_list[count]
            count += 1
    if count == 0:
        continue
    img /= count 
    # img /= len(files)

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{path}/{files[0].name}", img)
