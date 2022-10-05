# ################################################################################
# import os

# NAFNet_model = [
#     'NAFNet_charbonnier.yml',
#     'NAFNet_l1.yml',
#     'NAFNet_mse.yml',
#     'NAFNet_psnr.yml']

# for m in NAFNet_model:
#     os.system(f"python basicsr/test.py -opt inference/{m}")

# os.system('find ./results -name "*_gt.png" -exec rm {} \;')

# ################################################################################
# import os
# import random
# import shutil
# from pathlib import Path

# import cv2
# import numpy as np
# import torch
# from tqdm import tqdm

# random_seed = 10
# torch.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)
# torch.cuda.manual_seed(random_seed)


# file_list = list(Path('results').rglob('*.png'))
# keys = set(map(lambda x:x.name.split('_')[0], file_list))
# sorted_list = [[y for y in file_list if y.name.split('_')[0]==x] for x in keys]

# print(sorted_list[0])

# ensemble_list = ['test_']

# path = 'datasets/Dacon/test/lr_ensemble_concat'
# os.makedirs(path, exist_ok=True)

# # weights_list = [0.3, 1.8, 0.9]

# for files in tqdm(sorted_list):
#     concat_list = []
#     count = 0
#     for f in files:
#         if any(exp in str(f) for exp in ensemble_list):
#             if '_gt' in f.name:
#                 continue
#             temp = cv2.imread(str(f))
#             concat_list.append(temp)
#             count += 1
#     if not len(concat_list):
#         continue
#     concat_img = np.concatenate(concat_list, axis=2)
#     np.save(f"{path}/{files[0].stem}.npy", concat_img)

# os.system(f"python basicsr/test.py -opt inference/Deblur_ensemble.yml")

# path = 'results/test_Deblur_ensemble/visualization'
# os.makedirs(f"{path}/gt", exist_ok=True)
# os.system(f"mv {path}/custom/*_gt.png {path}/gt")

# ################################################################################
# import os
# import random
# import shutil
# from pathlib import Path

# import cv2
# import numpy as np
# import torch
# from tqdm import tqdm

# random_seed = 10
# torch.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)
# torch.cuda.manual_seed(random_seed)

# if Path('results/lr_ensemble').exists():
#     shutil.rmtree(Path('lr_results/ensemble'))

# file_list = list(Path('results').rglob('*.png'))
# keys = set(map(lambda x:x.name.split('_')[0], file_list))
# sorted_list = [[y for y in file_list if y.name.split('_')[0]==x] for x in keys]

# ensemble_list = [str(x) for x in Path('results').iterdir() if not x.is_file() and '1stage_' in str(x)]
# print(ensemble_list)

# path = 'results/lr_ensemble'
# os.makedirs(path, exist_ok=True)

# # weights_list = [0.3, 1.8, 0.9]

# for files in tqdm(sorted_list):
#     img = np.zeros((512,512,3))
#     count = 0
#     # weights_list = [1] * len(files)
#     for f in files:
#         if any(exp in str(f) for exp in ensemble_list):
#             if '_gt' in f.name:
#                 continue
#             temp = cv2.imread(str(f)).astype(np.float32)
#             img += temp # * weights_list[count]
#             count += 1
#     if count == 0:
#         continue
#     img /= count 
#     # img /= len(files)

#     img = np.clip(img, 0, 255).astype(np.uint8)
#     cv2.imwrite(f"{path}/{files[0].name}", img)


# ################################################################################
# import os

# HAT_model = [
#     'HAT_finetune_charbonnier.yml',
#     'HAT_finetune_l1.yml',
#     # 'HAT_finetune_mse.yml',
#     'HAT_finetune_psnr.yml',
#     # 'HAT_OuDD.yml',
#     # 'HAT-L_FDD.yml',
#     # 'HAT_Dacon.yml'
#     ]

# for m in HAT_model:
#     os.system(f"python basicsr/test.py -opt inference/{m}")

################################################################################
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

if Path('results/hr_ensemble').exists():
    shutil.rmtree(Path('results/hr_ensemble'))

file_list = list(Path('results').rglob('*.png'))
keys = set(map(lambda x:x.name.split('_')[0], file_list))
sorted_list = [[y for y in file_list if y.name.split('_')[0]==x] for x in keys]

# ensemble_list = [str(x) for x in Path('results').iterdir() if not x.is_file() and '2stage_' in str(x)]
ensemble_list = {
    # "2stage_HAT_Dacon":1,
    # "2stage_HAT_OuDD":1,
    # "2stage_HAT_finetune_mse":1,
    "2stage_HAT_finetune_l1":1,
    "2stage_HAT_finetune_charbonnier":1,
    "2stage_HAT_finetune_psnr":1,
}
weights_sum = sum(list(ensemble_list.values()))
print(ensemble_list)


path = 'results/hr_ensemble'
os.makedirs(path, exist_ok=True)

for files in tqdm(sorted_list):
    img = np.zeros((2048,2048,3))
    count = 0
    for f in files:
        if any(exp in str(f) for exp in ensemble_list.keys()):
            if str(f.parent.name) not in ['custom', 'ensemble']:
                continue
            temp = cv2.imread(str(f)).astype(np.float32)
            img += temp * ensemble_list[str(f.parents[2].name)]
            count += 1
    if count == 0:
        continue
    img /= weights_sum 

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{path}/{files[0].name}", img)
