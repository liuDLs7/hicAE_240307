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


def getnzc(ngene, file_path,):
    D = np.loadtxt(file_path)
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    # non_zero_in_all, contacts_in_all, non_zero_in_diags, contacts_in_diags
    matrix = np.triu(A)
    # 计算非零值的个数
    nzia = np.count_nonzero(matrix)
    # 计算所有非零值的和
    cia = np.sum(matrix)
    # 获取每个对角线中非零值的个数
    nzid = [np.count_nonzero(np.diagonal(matrix, offset=i)) for i in range(0, A.shape[1])]
    # 获取每个对角线上元素的和
    cid = [np.sum(np.diagonal(matrix, offset=i)) for i in range(0, A.shape[1])]
    return nzia, cia, nzid, cid

    

def main():
    # **********************************调参部分*******************************************
    dataset = 'Lee'

    chr_num = 23 if dataset in ['Ramani', '4DN', 'Lee'] else 20
    tshow = [1,3,5,7,8,9,10,20,30]
    diags = 100
    is_read = True
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
        cell_num = 626
    elif dataset == '4DN':
        chr_lens = [250, 244, 198, 192, 181, 172, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63, 49,
                    52, 156]
        cell_num = 4115
    elif dataset == 'Lee':
        chr_lens = [251, 245, 200, 193, 182, 173, 161, 148, 143, 137, 137, 135, 116, 108, 103, 92, 83, 80, 61, 65, 49,
                    52, 157]
        cell_num = 4234
    elif dataset == 'Collombet':
        chr_lens = [196, 183, 160, 157, 152, 150, 146, 130, 125, 131, 122, 121, 121, 125, 104, 99, 95, 91, 62, 171]
    elif dataset == 'Flyamer':
        chr_lens = [196, 182, 160, 156, 152, 150, 146, 130, 125, 130, 122, 121, 121, 125, 104, 99, 95, 91, 62, 167]
    else:
        assert 0, print('check dataset name!')

    if is_read:
        # 从JSON文件中加载数据
        with open(dataset + ".json", "r") as json_file:
            data = json.load(json_file)

        # 获取每个键对应的值
        nz_all = data["nz_all"]
        c_all = data["c_all"]
        nzd_all = data["nzd_all"]
        cd_all = data["cd_all"]
        diags = min(diags, len(nzd_all))

        # a=np.load(dataset+'.npy')
        # print(a)
        # exit(0)

        # all_len = 0.0
        # for chr_len in chr_lens:
        #     for i in range(1, chr_len+1):
        #         all_len += i

        # all_len = all_len * cell_num
        # print('非0值在整个上三角阵中所占比例={}/{}={}\n'.format(nz_all, all_len, nz_all/all_len))
        save1 = []
        save2 = []
        save3 = []
        for diag in range(1, diags+1):
            d_len = 0
            for chr_len in chr_lens:
                for i in range(chr_len, chr_len-diag, -1):
                    d_len += i
            
            if diag not in tshow:
                continue

            d_len = d_len * cell_num

            nzd_nd = sum(nzd_all[:diag])
            # print('最中心{}条对角线中非0值占比={}/{}={}\n'.format(diag,nzd_nd,d_len,round(nzd_nd/d_len,3)))
            save1.append(nzd_nd/d_len)
            # print('最中心{}条对角线中非0值占整个上三角阵中非0值的比例={}/{}={}\n'.format(diag,nzd_nd,nz_all,round(nzd_nd/nz_all,3)))
            save2.append(nzd_nd/nz_all)

            c_nd = sum(cd_all[:diag])
            print('最中心{}条对角线中包含的接触数在总体中的占比={}/{}={}\n'.format(diag,c_nd,c_all,round(c_nd/c_all,3)))
            save3.append(c_nd/c_all)

        all_save=[save1,save2,save3]
        # print(all_save)
        all_save = np.array(all_save)
        # np.save(dataset+'.npy', all_save)

        exit(0)

    
    print(chr_lens)

    min_len = min(chr_lens)

    nz_all = 0.0
    c_all = 0.0
    nzd_all =  [0 for _ in range(min_len)]
    cd_all =  [0 for _ in range(min_len)]

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        target_sub_dir = os.path.join(target_dir, sub_dir)
        file_names = os.listdir(sub_path)
        nz_sub = 0.0
        c_sub = 0.0
        nzd_sub = [0 for _ in range(min_len)]
        cd_sub = [0 for _ in range(min_len)]

        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else chr_num
            ngene = chr_lens[chromosome_number - 1]
            non_zero_in_all, contacts_in_all, non_zero_in_diags, contacts_in_diags = getnzc(ngene=ngene, file_path=file_path)
            nz_sub += non_zero_in_all
            c_sub += contacts_in_all
            nzd_sub = [x + y for x, y in zip(nzd_sub[:min_len], non_zero_in_diags[:min_len])]
            cd_sub = [x + y for x, y in zip(cd_sub[:min_len], contacts_in_diags[:min_len])]

        print(sub_dir + ' has been processed!')
        nz_all += nz_sub
        c_all += c_sub
        nzd_all = [x + y for x, y in zip(nzd_sub[:min_len], nzd_all[:min_len])]
        cd_all = [x + y for x, y in zip(cd_sub[:min_len], cd_all[:min_len])]

    # 打印这四个值
    print("nz_all:", nz_all)
    print("c_all:", c_all)
    print("nzd_all:", nzd_all)
    print("cd_all:", cd_all)

    # 将这四个值保存到一个JSON文件中
    data = {
        "nz_all": nz_all,
        "c_all": c_all,
        "nzd_all": nzd_all,
        "cd_all": cd_all
    }

    with open(dataset + ".json", "w") as json_file:
        json.dump(data, json_file)


if __name__ == '__main__':
    main()
