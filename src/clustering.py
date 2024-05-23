from sklearn.cluster import \
    KMeans, AgglomerativeClustering, SpectralClustering, Birch
from pyspark.ml.clustering import BisectingKMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, NMF
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
from DIANA import DIANA
sys.path.append('../aenets')
from net import AE, AE2, Mish, AET
import os
import unittest
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import ClusterEnsembles as CE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, fowlkes_mallows_score



def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

# schicluster
def neighbor_ave_gpu(A, pad):
    if pad == 0:
        return torch.from_numpy(A).float().cuda()
    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll).cuda()
    B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad * 2)
    return B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)


def random_walk_gpu(A, rp):
    ngene, _ = A.shape
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag(torch.sum(A, 0) == 0).float()

    P = torch.div(A, torch.sum(A, 0))
    Q = torch.eye(ngene).cuda()
    I = torch.eye(ngene).cuda()
    for i in range(30):
        Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
        delta = torch.norm(Q - Q_new, 2)
        Q = Q_new
        if delta < 1e-6:
            break
    return Q


read_file_time = 0.0


def impute_gpu(ngene, pad, rp, file_path, reconstructed_data):
    global read_file_time
    t = time.time()
    D = np.loadtxt(file_path)
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    
    A = A + A.T + 1

    A = np.log2(A)

    A = neighbor_ave_gpu(A, pad)
    if rp == -1:
        Q = A[:]
    else:
        Q = random_walk_gpu(A, rp)
    return Q



def base2ensemble(bd, mode: int, nc, limit):
    file_names1 = os.listdir(bd)
    file_names = []
    for file_name in file_names1:
        if limit in file_name:
            file_names.append(file_name)
    
    base_clusters = []
    for file_name in file_names:
        print(file_name)
        file_path = os.path.join(bd, file_name)
        base_clusters.append(list(np.load(file_path)))
    base_clusters = np.array(base_clusters)
    solvers = ['cspa', 'hbgf', 'hgpa', 'mcla', 'nmf', 'all']
    label_pred = CE.cluster_ensembles(base_clusters, solver=solvers[mode], nclass=nc, random_state=42)
    return label_pred, solvers[mode]


def run_on_model(origin_dir,model_dir, train_epochs, network, ngenes, nc, model_type, ndim=20, is_X=False, prct=20, clusters=None, is_code='True'):
    #assert cluster in ['k-means', 'diana', 'agg', 'spec', 'ensemble'], print('no such cluster!')
    assert clusters is not None and clusters != [], print('no cluster!')
    matrix = []
    ndims = []
    for c, ngene in enumerate(ngenes):
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        model_path = os.path.join(model_dir, 'chr' + c + '_' + str(train_epochs) + 'epochs.pth')

        with open(os.path.join(model_dir, 'chr' + c + '_datasize.json'), 'r') as json_file:
            d = json.load(json_file)
            ipt_size = d['ipt']
            opt_size = d['opt']

        # 创建模型实例
        if model_type == 'AE':
            model = AE(ipt_size, opt_size)
        elif model_type == 'AE2':
            model = AE2(ipt_size, opt_size)
        elif model_type == 'AET':
            model_type = AET(ipt_size, opt_size)
        else:
            print('check model name!')

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)

        Q_concat = []
        with torch.no_grad():
            for cell in network:
                file_name = cell + '_chr' + c + '.npy'
                # last_slash_index = file_name.rfind('/')
                # if last_slash_index != -1:  # Ensure that at least one '/' is found
                #     extracted_content = file_name[last_slash_index + 1:-4]
                # txt_name = extracted_content + '.txt'
                # txt_path = os.path.join(origin_dir, txt_name)
                # x = self.encoder(x)
                # x = self.bn1(x)
                # x= self.relu(x)
                # x = self.decoder(x)
                # x = self.bn2(x)
                # x = self.sigmoid(x)

                test_data = np.load(file_name)
                # Q_concat.append(test_data)
                test_data = torch.from_numpy(test_data).to(device)
                if model_type == 'AE':
                    embedding = model.encoder(test_data)
                    reconstructed_datas = model.decoder(embedding)
                elif model_type == 'AE2':
                    data = model.encoder(test_data)
                elif model_type == 'AET':
                    test_data = model.encoder(test_data)
                    test_data = test_data.unsqueeze(1)  # 添加时间维度以适应Transformer输入
                    data = model.transformer_encoder(test_data)
                
                # Q_concat.append(test_data.cpu().numpy())
                if is_code:
                    if model_type == 'AE':
                        Q_concat.append(embedding.cpu().numpy())
                    elif model_type == 'AE2':
                        embedding = model.relu(data)
                        Q_concat.append(embedding.cpu().numpy())
                    elif model_type == 'AET':
                        embedding = model.relu(data)
                        Q_concat.append(embedding.cpu().numpy())
                else:
                    if model_type == 'AE':
                        Q_concat.append(reconstructed_datas.cpu().numpy())
                    elif model_type == 'AE2':
                        data = model.relu(data)
                        data = model.decoder(data)
                        c = model.sigmoid(data)
                        Q_concat.append(reconstructed_datas.cpu().numpy())
                    elif model_type == 'AET':
                        data = data.squeeze(1)  # 移除时间维度
                        data = model.decoder(data)
                        reconstructed_datas = torch.sigmoid(data)


        Q_concat = np.array(Q_concat)

        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])

        ndim = int(min(Q_concat.shape) * 0.2) - 1
        ndims.append(ndim)
        print(Q_concat.shape)
        # pca = PCA(n_components=ndim)
        # R_reduce = pca.fit_transform(Q_concat)
        # matrix.append(R_reduce)
        matrix.append(Q_concat)
    matrix = np.concatenate(matrix, axis=1)
    print(matrix.shape)
    ndim = min(ndims)
    #pca = PCA(n_components=ndim)

    # 正常步骤
    # nmf = NMF(n_components=ndim ,init='nndsvd', max_iter=5000)
    # matrix_reduce = nmf.fit_transform(matrix)
    pca = PCA(n_components=int(min(matrix.shape) * 0.2) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    matrix_reduce = matrix_reduce[:, :ndim]

    # 消融实验
    # matrix_reduce =matrix

    print('ndim = ' + str(ndim))
    print(matrix_reduce.shape)
    # 下载到PC端可视化
    # np.save('matrix_reduced.npy', matrix_reduce)
    tsne = TSNE(learning_rate=1000, random_state=42)
    matrix_tsne = tsne.fit_transform(matrix_reduce)

    labels = []
    for cluster in clusters:
        if 'k-means' == cluster:
            # k-means
            #kmeans = KMeans(n_clusters=nc, n_init=500).fit(matrix_tsne)
            kmeans = KMeans(n_clusters=nc, n_init=500, random_state=42).fit(matrix_reduce)
            print('n_iter =', kmeans.n_iter_)
            labels.append(kmeans.labels_)

        # elif 'gauss' == cluster:
        #     gmm = GaussianMixture(n_components=nc, covariance_type='diag', random_state=42)
        #     labels.append(gmm.fit_predict(matrix_tsne))

        # elif cluster == 'diana':
        #     diana = DIANA(nclusters=nc, random_state=None)
        #     tsne = TSNE(learning_rate=1000, random_state=None)
        #     matrix_tsne = tsne.fit_transform(matrix_reduce)
        #     labels.append(diana.fit_predict(matrix_tsne))

        elif cluster == 'agg':
            # 层次聚类（Agglomerative Clustering）
            distances = ["euclidean", "l1", "l2", "manhattan", "cosine"]
            links = ['ward', 'average', 'complete']
            # 这组参数可以用?
            # agg_clustering = AgglomerativeClustering(n_clusters=nc, affinity=distances[4], linkage=links[2])
            agg_clustering = AgglomerativeClustering(n_clusters=nc)
            labels.append(agg_clustering.fit_predict(matrix_reduce))
            # labels = agg_clustering.fit_predict(matrix_tsne)

        elif cluster== 'biskmeans':
            bisect_means = BisectingKMeans(k=nc, maxIter=50, minDivisibleClusterSize=1.0)
            print(bisect_means.fit(matrix_reduce).predict)

        elif cluster == 'agg2':
            # 层次聚类（Agglomerative Clustering）
            distances = ["euclidean", "l1", "l2", "manhattan", "cosine"]
            links = ['ward', 'average', 'complete']
            # 这组参数可以用?
            agg_clustering = AgglomerativeClustering(n_clusters=nc, affinity=distances[4], linkage=links[2])
            # agg_clustering = AgglomerativeClustering(n_clusters=nc)
            # labels = agg_clustering.fit_predict(matrix_reduce[:, :ndim])
            labels.append(agg_clustering.fit_predict(matrix_tsne))

        elif cluster == 'birch':
            # 平衡迭代
            birch = Birch(n_clusters=nc, threshold=1, branching_factor=50)
            labels.append(birch.fit_predict(matrix_reduce))

        elif cluster == 'spec':
            # 谱聚类（Spectral Clustering）
            spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100, random_state=42)
            labels.append(spectral_clustering.fit_predict(matrix_reduce))

        elif cluster == 'mcla':
            mode_id = 3
            solvers = ['cspa', 'hbgf', 'hgpa', 'mcla', 'nmf', 'all']
            labels.append(CE.cluster_ensembles(np.transpose(matrix_reduce), solver=solvers[mode_id], nclass=nc, random_state=42))

        # elif cluster == 'specg':
        #     # 谱图聚类（Spectral Clustering）
        #     # 构建图的邻接矩阵（这里以K近邻图为例）
        #     k = 10
        #     connectivity = kneighbors_graph(matrix_reduce, n_neighbors=k, include_self=False)

        #     # 谱图聚类（Spectral Graph Clustering）
        #     spectral_clustering = SpectralClustering(n_clusters=nc, affinity='precomputed', assign_labels='kmeans', n_init=100)
        #     labels.append(spectral_clustering.fit_predict(connectivity))

        else:
            print('check {} cluster!'.format(cluster))

    return labels, matrix_reduce


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)

    # *******************************调参部分*****************************************

    dataset = 'Lee'
    sdir = 'diag8'
    extra = 'm20_o6'
    train_epochs = 500
    prct = 30
    # k-means, agg, agg2, spec, ensemble, birch
    clusters = ['k-means', 'agg', 'agg2', 'spec', 'birch']
    is_code: bool = True
    extra2 = '' if not is_code else '_code'

    is_save = True
    # 是否使用所有获得的标签
    is_all = True

    model_type = 'AE'
    extra3 = ''
    # 包含limit中内容的才会被用于最后的聚类
    limit = ''

    # ********************************************************************************

    # 保存单种聚类方法结果，以便聚类集成使用
    cluster_dir = '../PC_datas/predict_labels/{}'.format(dataset)
    os.makedirs(cluster_dir, exist_ok=True)
    cluster_paths = []
    for cluster in clusters:
        cluster_file = '{}_{}_{}_{}_{}{}{}.npy'.format(sdir, extra, train_epochs, prct, cluster, extra2, extra3)
        cluster_path = os.path.join(cluster_dir, cluster_file)
        cluster_paths.append(cluster_path)

    # 分类数
    if dataset == 'Lee':
        nc = 14
    elif dataset == '4DN':
        nc = 5
    elif dataset == 'Ramani':
        nc = 4
    elif dataset == 'Collombet':
        nc = 5
    elif dataset == 'Flyamer':
        nc = 3
    else:
        assert 0, print('check dataset name!')

    ndim = 20
    # 含X染色体总数
    chr_num = 23 if dataset in ['Ramani', '4DN', 'Lee'] else 20
    is_X = False if dataset == 'Lee' else True

    # 加载数据位置+
    origin_dir = '../../../Downloads/CTPredictor/Data_filter/{}'.format(dataset)
    root_dir = '../../Datas/vectors/{}/{}'.format(dataset, sdir)
    data_info_path = os.path.join(root_dir, 'data_info.json')

    # 模型保存文件
    model_dir = '../../models/{}/{}{}'.format(dataset, sdir, extra)
    print('model_dir=', model_dir)

    # 加载ngenes
    with open(data_info_path, 'r') as f:
        ngenes = json.load(f)['chr_lens']

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
            cell_path = os.path.join(sub_path, 'cell_' + str(i))
            network.append(cell_path)
            y.append(str2dig[label_dir])

    index_array = np.arange(len(y))
    random.shuffle(index_array)

    # 打乱y和network
    y_shuffled = [y[i] for i in index_array]
    network_shuffled = [network[i] for i in index_array]

    if train_epochs <= 0:
        assert 0, 'error!'
    else:
        cluster_labels, mr = run_on_model(origin_dir=origin_dir, model_dir=model_dir, train_epochs=train_epochs, network=network_shuffled, 
                                      ngenes=ngenes, nc=nc, model_type=model_type, ndim=ndim, is_X=is_X, prct=prct, 
                                      clusters=clusters, is_code=is_code)
        #cluster_labels = np.load(cluster_path)
     
    y = np.array(y)
    cluster_labels_sorted = []
    for cluster_label in cluster_labels:
        cluster_labels_restored = [x for _, x in sorted(zip(index_array, cluster_label))]
        cluster_labels_sorted.append(np.array(cluster_labels_restored))

    for cluster, cluster_label, cluster_path in zip(clusters, cluster_labels_sorted, cluster_paths):
        # 计算调整兰德指数和归一化互信息
        anmi_score = adjusted_mutual_info_score(y, cluster_label)
        ari = adjusted_rand_score(y, cluster_label)
        nmi = normalized_mutual_info_score(y, cluster_label)
        hm = homogeneity_score(y, cluster_label)
        fm = fowlkes_mallows_score(y, cluster_label)

        # print("ANMI Score:", anmi_score)
        print('cluster =', cluster)
        print("Adjusted Rand Index (ARI):", round(ari,3))
        print("Normalized Mutual Information (NMI):", round(nmi,3))
        print("Homogeneity (HM):", round(hm,3))
        print("Fowlkes_Mallows (FM):", round(fm,3))

        if is_save:
            np.save('../PC_datas/labels/{}_labels'.format(dataset), y)
            np.save(cluster_path, cluster_label)
            print('The result has been saved in {}'.format(cluster_path))
        
        print('-' * 50)

    print('root_dir={}\nmodel_dir={}\ncluster={}\nnc={}\nprct={}\ntrain_epochs={}'.format(root_dir, model_dir, clusters,
                                                                                          nc, prct,
                                                                                          train_epochs))

    # 集成
    print('-'*50)
    mode_id = 3
    solvers = ['cspa', 'hbgf', 'hgpa', 'mcla', 'nmf', 'all']
    label_true = y
    if is_all:
        print('-'*50)
        # 先输出当次结果
        ensemble_labels = CE.cluster_ensembles(np.array(cluster_labels_sorted), solver=solvers[mode_id], nclass=nc, random_state=42)
        ari = adjusted_rand_score(label_true, ensemble_labels)
        nmi = normalized_mutual_info_score(label_true, ensemble_labels)
        hm = homogeneity_score(label_true, ensemble_labels)
        fm = fowlkes_mallows_score(label_true, ensemble_labels)

        print("Adjusted Rand Index (ARI):", round(ari,3))
        print("Normalized Mutual Information (NMI):", round(nmi,3))
        print("Homogeneity (HM):", round(hm,3))
        print("Fowlkes_Mallows (FM):", round(fm,3))
        print('-'*50)
        
        #再输出全部结果
        ensemble_labels, mode = base2ensemble(cluster_dir, mode_id, nc, limit)

        # 用于可视化
        mr_sorted = [x for _, x in sorted(zip(index_array, mr))]
        np.save('{}.npy'.format(dataset), mr_sorted)
        np.save('{}_label.npy'.format(dataset), ensemble_labels)

    else:
        ensemble_labels = CE.cluster_ensembles(np.array(cluster_labels_sorted), solver=solvers[mode_id], nclass=nc, random_state=42)
        mode = solvers[mode_id]

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(label_true, ensemble_labels)
    nmi = normalized_mutual_info_score(label_true, ensemble_labels)
    hm = homogeneity_score(label_true, ensemble_labels)
    fm = fowlkes_mallows_score(label_true, ensemble_labels)

    print("Adjusted Rand Index (ARI):", round(ari,3))
    print("Normalized Mutual Information (NMI):", round(nmi,3))
    print("Homogeneity (HM):", round(hm,3))
    print("Fowlkes_Mallows (FM):", round(fm,3))
    print('mode=', mode)

