import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MyDataset
import os
import time
import numpy as np
np.random.seed(42)
import sys
from tqdm import tqdm as progress_bar
from tqdm import trange
import re

sys.path.append('../aenets')
from net import AE, AE2layers


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def make_datasets(root_dir, network, update_mask, mask_rate):
    Q_concat = []
    for file_path in network:
        file_data = np.load(file_path)
        Q_concat.append(file_data)

    dataset = MyDataset(root_dir=root_dir, Q_concat=Q_concat, file_names=network, is_mask=True, random_mask=True,
                        mask_rate=mask_rate, update_mask=update_mask, is_shuffle=True)
    return dataset


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(device)

    # *******************************调参部分*****************************************

    ds = 'Lee_merged'
    sd = 'diag8'
    extra = 'm10_o6'

    # 是否使用训练过的模型继续训练
    is_pretrained = False
    if is_pretrained:
        load_epochs = 500
    else:
        load_epochs = 'None'
    save_epochs = 500

    batch_size = 256
    lr = 1e-3
    update_mask = True
    mask_rate = 0.10
    # 用来调整embedding层大小
    opt_rate = 1.0 / 6.0

    model_name = 'AE'

    # ********************************************************************************

    # 加载数据位置
    root_dir = '../../Datas/vectors/{}/{}'.format(ds, sd)
    data_info_path = os.path.join(root_dir, 'data_info.json')

    # 模型保存文件
    model_dir = '../../models/{}/{}_{}_{}'.format(ds, sd, model_name, extra)
    os.makedirs(model_dir, exist_ok=True)

    label_dirs = get_subdirectories(root_dir)
    if 'masks' in label_dirs:
        label_dirs.remove('masks')

    network = []

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        file_names = os.listdir(sub_path)
        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            network.append(file_path)

    train_dataset = make_datasets(root_dir=root_dir, network=network, update_mask=update_mask, mask_rate=mask_rate)

    start_time = time.time()

    load_model_path = os.path.join(model_dir, str(load_epochs) + 'epochs.pth')
    save_model_path = os.path.join(model_dir, str(save_epochs) + 'epochs.pth')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ipt_size = train_dataset.datasize
    opt_size = int(min(len(train_dataset), ipt_size) * opt_rate) - 1

    size_data = {
        'ipt': ipt_size,
        'opt': opt_size
    }

    with open(os.path.join(model_dir, 'datasize.json'), 'w') as f:
        json.dump(size_data, f)

    # 创建模型实例并将其移动到GPU上
    if model_name == 'AE':
        model = AE(ipt_size, opt_size)
    elif model_name == 'AE2layers':
        model = AE2layers(ipt_size, opt_size)
    else:
        assert 0, print('wrong model name!')
    if is_pretrained:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    num_epochs = save_epochs - load_epochs if is_pretrained else save_epochs
    start = time.time()
    for epoch in trange(num_epochs):
        # train_dataset.gen_mask_time = 0.0
        # train_dataset.read_dic_time = 0.0
        # print(f"start Epoch [{epoch + 1}/{num_epochs}]")
        running_loss = 0.0  # 用于累积整个训练集上的损失值
        for train_data in train_loader:
            datas, _ = train_data
            if isinstance(datas, list):
                # 此时datas是由[original_datas,masked_datas]组成
                original_datas = datas[0]
                masked_datas = datas[1]
            else:
                original_datas = datas
                masked_datas = datas

            original_datas = original_datas.view(original_datas.size(0), -1).to(device)
            masked_datas = masked_datas.view(masked_datas.size(0), -1).to(device)

            reconstructed_datas = model(masked_datas)
            loss = criterion(reconstructed_datas, original_datas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 手动释放内存
            # del original_datas
            # del masked_datas
            # torch.cuda.empty_cache()

            running_loss += loss.item() * original_datas.size(0)  # 累积损失值

        epoch_loss = running_loss / len(train_loader.dataset)  # 计算整个训练集上的平均损失值
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss * 1e4:.4f}")

    print('complete train!')
    print(' use time: ' + str(time.time() - start))
    # print('read_dic_time: ' + str(train_dataset.read_dic_time))
    # print('gen_mask_time: ' + str(train_dataset.gen_mask_time))
    # 保存模型
    print('saving model...')
    torch.save(model.state_dict(), save_model_path)
    print('model saved!')
    # time.sleep(5)

    print('root_dir={}\nmodel_dir={}\nload_epochs={}\nsave_epochs={}\nbatch_size={}\nlr={}\nupdate_mask={}\nmask_rate={}\nopt_rate={}'.format(
            root_dir, model_dir, load_epochs, save_epochs, batch_size, lr, update_mask, mask_rate, opt_rate))
    print('model=', model_name)
