from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from dataset import MyDataset
import torch
import numpy as np

np.random.seed(42)
from torch.utils.data import DataLoader
import time
import random
import os
import json
from scipy.sparse import csr_matrix
import re
import sys
from sklearn.manifold import TSNE
# from DIANA import DIANA

sys.path.append('../aenets')
from net import AE, AE2layers


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def run_on_model(model_dir, train_epochs, network, nc, model_name, ndim=20,  prct=20, cluster='k-means'):
    assert cluster in ['k-means', 'diana', 'agg', 'spec', 'ensemble'], print('no such cluster!')
    matrix = []

    model_path = os.path.join(model_dir, str(train_epochs) + 'epochs.pth')

    with open(os.path.join(model_dir, 'datasize.json'), 'r') as json_file:
        d = json.load(json_file)
        ipt_size = d['ipt']
        opt_size = d['opt']

    # 创建模型实例
    if model_name == 'AE':
        model = AE(ipt_size, opt_size)
    elif model_name == 'AE2layers':
        model = AE2layers(ipt_size, opt_size)
    else:
        assert 0, print('wrong model name!')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    Q_concat = []
    with torch.no_grad():
        for file_name in network:
            test_data = np.load(file_name)
            test_data = torch.from_numpy(test_data).to(device)
            embedding = model.encoder(test_data)
            reconstructed_datas = model.decoder(embedding)

            # Q_concat.append(test_data)
            # Q_concat.append(embedding.cpu().numpy())
            Q_concat.append(reconstructed_datas.cpu().numpy())

    Q_concat = np.array(Q_concat)

    if prct > -1:
        thres = np.percentile(Q_concat, 100 - prct, axis=1)
        Q_concat = (Q_concat > thres[:, None])

    ndim = int(min(Q_concat.shape) * 0.2) - 1
    print(Q_concat.shape)
    pca = PCA(n_components=ndim)
    matrix_reduce = pca.fit_transform(Q_concat)
    print('ndim = ' + str(ndim))
    print(matrix_reduce.shape)
    # 下载到PC端可视化
    # np.save('matrix_reduced.npy', matrix_reduce)

    if cluster == 'k-means':
        # k-means
        kmeans = KMeans(n_clusters=nc, n_init=500).fit(matrix_reduce[:, :ndim])
        print('n_iter =', kmeans.n_iter_)
        labels = kmeans.labels_

    # elif cluster == 'diana':
    #     diana = DIANA(nclusters=nc, random_state=None)
    #     tsne = TSNE(learning_rate=1000, random_state=None)
    #     matrix_tsne = tsne.fit_transform(matrix_reduce[:, :ndim])
    #     labels = diana.fit_predict(matrix_tsne)

    elif cluster == 'agg':
        # 层次聚类（Agglomerative Clustering）
        distances = ["euclidean", "l1", "l2", "manhattan", "cosine"]
        links = ['ward', 'average', 'complete']
        # 这组参数可以用?
        # agg_clustering = AgglomerativeClustering(n_clusters=nc, affinity=distances[4], linkage=links[2])
        agg_clustering = AgglomerativeClustering(n_clusters=nc)
        labels = agg_clustering.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'spec':
        # 谱聚类（Spectral Clustering）
        spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        labels = spectral_clustering.fit_predict(matrix_reduce[:, :ndim])

    # elif cluster == 'ensemble':
    # kmeans = KMeans(n_clusters=nc, n_init=500)
    # agg_clustering = AgglomerativeClustering(n_clusters=nc)
    # spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
    # ensemble_model = ClusteringEnsemble(models=[kmeans, agg_clustering, spectral_clustering])
    # ensemble_model.fit(matrix_reduce[:, :ndim])
    # labels = ensemble_model.predict(matrix_reduce[:, :ndim])

    else:
        print('报错也能出问题？再改改！')
        exit(1)

    return labels


def run_original_data(network, ngenes, nc, ndim=20, is_X=False, prct=20):
    matrix = []
    for c, ngene in enumerate(ngenes):
        labels = []
        file_names = []
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        start_time = time.time()
        Q_concat = []
        for cell in network:
            file_name = cell + '_chr' + c + '.npy'
            file_names.append(file_name)
            Q_concat.append(np.load(file_name))

        Q_concat = np.array(Q_concat)

        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])

        ndim = int(min(Q_concat.shape) * 0.2) - 1
        print(Q_concat.shape)
        # U, S, V = torch.svd(Q_concat, some=True)
        # R_reduce = torch.mm(U[:, :ndim], torch.diag(S[:ndim])).cuda().numpy()
        pca = PCA(n_components=ndim)
        R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    pca = PCA(n_components=min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    # ndim = 30
    print('ndim = ' + str(ndim))
    kmeans = KMeans(n_clusters=nc, n_init=500, init='k-means++').fit(matrix_reduce[:, :ndim])
    return kmeans.labels_


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)

    # *******************************调参部分*****************************************

    dataset = 'Ramani_merged'
    sdir = 'diag8'
    extra = 'm10_o6'
    train_epochs = 500
    prct = 30
    # k-means, agg, spec, ensemble
    cluster = 'spec'
    model_name = 'AE'

    is_save = False

    # ********************************************************************************

    # 保存单种聚类方法结果，以便聚类集成使用
    cluster_dir = '../PC_datas/predict_labels/{}'.format(dataset)
    cluster_file = '{}_{}_{}_{}_{}_{}.npy'.format(model_name, sdir, extra, train_epochs, prct, cluster)
    cluster_path = os.path.join(cluster_dir, cluster_file)
    os.makedirs(cluster_dir, exist_ok=True)

    # 分类数
    if 'Lee' in dataset:
        nc = 14
    elif '4DN' in dataset:
        nc = 5
    elif 'Ramani' in dataset:
        nc = 4
    else:
        assert 0, print('check dataset name!')

    ndim = 20
    # 含X染色体总数
    chr_num = 23
    is_X = False if dataset == 'Lee' else True

    # 加载数据位置
    root_dir = '../../Datas/vectors/{}/{}'.format(dataset, sdir)
    data_info_path = os.path.join(root_dir, 'data_info.json')

    # 模型保存文件
    model_dir = '../../models/{}/{}_{}_{}'.format(dataset, sdir, model_name, extra)
    print('model_dir=', model_dir)

    label_dirs = get_subdirectories(root_dir)
    if 'masks' in label_dirs:
        label_dirs.remove('masks')
    y = []
    network = []

    str2dig = {}
    x = []

    for i, label_name in enumerate(label_dirs):
        str2dig[label_name] = i

    print(str2dig)

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        file_names = os.listdir(sub_path)
        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            network.append(file_path)
            y.append(str2dig[label_dir])

    index_array = np.arange(len(y))
    random.shuffle(index_array)

    # 打乱y和network
    y_shuffled = [y[i] for i in index_array]
    network_shuffled = [network[i] for i in index_array]

    if train_epochs <= 0:
        cluster_labels = run_original_data(network_shuffled, nc, ndim, is_X, prct)
    else:
        cluster_labels = run_on_model(model_dir=model_dir, train_epochs=train_epochs, network=network_shuffled, nc=nc,
                                      model_name=model_name, ndim=ndim, prct=prct, cluster=cluster)
        # cluster_labels = np.load(cluster_path)

    y = np.array(y)
    cluster_labels_restored = [x for _, x in sorted(zip(index_array, cluster_labels))]
    cluster_labels = np.array(cluster_labels_restored)

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    hm = homogeneity_score(y, cluster_labels)
    fm = completeness_score(y, cluster_labels)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("Homogeneity (HM):", hm)
    print("Completeness (FM):", fm)

    print('root_dir={}\nmodel_dir={}\ncluster={}\nnc={}\nprct={}\ntrain_epochs={}'.format(root_dir, model_dir, cluster,
                                                                                          nc, prct,
                                                                                          train_epochs))

    if is_save:
        np.save('../PC_datas/labels/{}_labels'.format(dataset), y)
        np.save(cluster_path, cluster_labels)
        print('The result has been saved in {}'.format(cluster_path))
