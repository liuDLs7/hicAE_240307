import datetime
import json
import re
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def main():
    # **********************************调参部分*******************************************
    dataset = 'Lee'

    chr_num = 23 if dataset in ['Ramani', '4DN', 'Lee'] else 20
    diags = 8
    is_read = False
    extra = ''
    # ************************************************************************************

    root_dir = '../../../Downloads/CTPredictor/Data_filter/{}'.format(dataset)
    target_dir = '../../Datas/sparsity'
    processed_dir = '../../Datas/{0}/{0}_processed'.format(dataset)

    sub_dirs = get_subdirectories(root_dir)

    # 定义需要排除的元素
    exclude_elements = ['avgs', 'weights']

    for sub_dir in sub_dirs:
        if sub_dir in exclude_elements:
            sub_dirs.remove(sub_dir)

    # chr_lens = get_max_chr_len(processed_dir, chr_num=chr_num)
    if dataset == 'Ramani':
        chr_lens = [250, 244, 198, 192, 181, 171, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63, 49,
                    52, 155]
    elif dataset == '4DN':
        chr_lens = [250, 244, 198, 192, 181, 172, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63, 49,
                    52, 156]
    elif dataset == 'Lee':
        chr_lens = [251, 245, 200, 193, 182, 173, 161, 148, 143, 137, 137, 135, 116, 108, 103, 92, 83, 80, 61, 65, 49,
                    52, 157]
    elif dataset == 'Collombet':
        chr_lens = [196, 183, 160, 157, 152, 150, 146, 130, 125, 131, 122, 121, 121, 125, 104, 99, 95, 91, 62, 171]
    elif dataset == 'Flyamer':
        chr_lens = [196, 182, 160, 156, 152, 150, 146, 130, 125, 130, 122, 121, 121, 125, 104, 99, 95, 91, 62, 167]
    else:
        assert 0, print('check dataset name!')

    cell_all = 0

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        target_sub_dir = os.path.join(target_dir, sub_dir)
        file_names = os.listdir(sub_path)
        chr_sub = len(file_names)
        cell_sub = chr_sub / 23
        cell_all += cell_sub
        print('{} has {} cells'.format(sub_dir, str(int(cell_sub))))

    print('There has {} cells in {}'.format(str(int(cell_all)), dataset))

        


if __name__ == '__main__':
    main()
