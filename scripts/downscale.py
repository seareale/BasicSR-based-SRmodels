import os
import random
import sys
from configparser import Interpolation
from multiprocessing import Pool
from os import path as osp

import cv2
import numpy as np
sys.path.append('.')
from basicsr.utils import scandir
from tqdm import tqdm

# Random seed
import torch
random_seed = 10
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(random_seed)
import random
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


def main():
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HR images
    opt['input_folder'] = 'datasets/Dacon/train/hr'
    opt['save_folder'] = 'datasets/Dacon/train/lr_random'
    opt['resize'] = 0.25
    opt['offset'] = None
    extract_lrimages(opt)

    # # HR images
    # opt['input_folder'] = 'datasets/Dacon/train/hr'
    # opt['save_folder'] = 'datasets/Dacon/train/lr_bicubic'
    # opt['resize'] = 0.25
    # opt['offset'] = None
    # extract_lrimages(opt)

def extract_lrimages(opt):
    """Crop images to subimages.
    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    # if not osp.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)
    print(f'{input_folder} | mkdir {save_folder} ...')
    # else:
    #     print(f'Folder {save_folder} already exists. Exit.')
    #     sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

def worker(path, opt):
    """Worker for each process.
    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        resize (int): Resize.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.
    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    resize = opt['resize']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h,w,_ = img.shape
    h_, w_ = int(h*resize), int(w*resize)

    # img = cv2.resize(img, (w_,h_), interpolation=cv2.INTER_AREA)
    
    iter = random.randint(1,5)
    for idx in range(0, iter):
        img = cv2.resize(img, (w_,h_), interpolation=random.randint(0,4))
        if idx == iter - 1:
            break
        img = cv2.resize(img, (w,h), interpolation=random.randint(0,4))
    cv2.imwrite(
                osp.join(opt['save_folder'], f"{opt['offset']+'_' if opt['offset'] else ''}{img_name}{extension}"), img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img.shape} {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
