import numpy as np
import os
import re


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


if __name__ == '__main__':
    # **********************************调参部分*******************************************
    dataset = 'Lee'
    pad = 0
    rp = -1
    mode = 'chr_max'
    process_pattern = 'diag'
    m = 8
    chr_num = 23
    extra = ''

    is_X = False if dataset == 'Lee' else True
    # ************************************************************************************

    root_dir = '../../Datas/vectors/{0}/{1}{2}{3}'.format(
        dataset, process_pattern, 'all' if m == -1 else str(m), extra)
    target_dir = '../../Datas/vectors/{0}_merged/{1}{2}{3}'.format(
        dataset, process_pattern, 'all' if m == -1 else str(m), extra)

    sub_dirs = get_subdirectories(root_dir)

    # 定义需要排除的元素
    exclude_elements = ['avgs', 'weights', 'masks']

    for sub_dir in sub_dirs:
        if sub_dir in exclude_elements:
            sub_dirs.remove(sub_dir)
    
    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        sub_tpath = os.path.join(target_dir, sub_dir)
        os.makedirs(sub_tpath, exist_ok=True)
        files = os.listdir(sub_path)
        file_num = 0
        cell_numbers = []
        for file in files:
            file_num += 1
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).npy', file)
            cell_number = int(match.group(1))
            if cell_number not in cell_numbers:
                cell_numbers.append(cell_number)
        cell_num = int(file_num / chr_num)
        # for i in range(1, cell_num + 1):
        #     cell_path = os.path.join(sub_path, 'cell_' + str(i))
        #     network.append(cell_path)
        #     y.append(str2dig[label_dir])
        for i in cell_numbers:
            merged_cell = []
            cell_path = os.path.join(sub_tpath, 'cell' + str(i) + '.npy')
            for j in range(chr_num):
                c = 'X' if is_X and j == chr_num - 1 else str(j + 1)
                chr_name = 'cell_' + str(i) + '_chr' + c + '.npy'
                chr_path = os.path.join(sub_path, chr_name)
                merged_cell.append(np.load(chr_path))
            merged_cell = np.concatenate(merged_cell)
            np.save(cell_path, merged_cell)

        print(sub_dir, 'has been merged')

    print('merged cells have been saved in ', target_dir)

