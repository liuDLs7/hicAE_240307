from sklearn.cluster import \
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch, MiniBatchKMeans, MeanShift
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture

from dataset import MyDataset
import torch
import numpy as np
import time
import random
import os
import json
from scipy.sparse import csr_matrix
import re
import sys
#from clusters import ClusteringEnsemble, EnsembleClustering
from sklearn.ensemble import VotingClassifier
from collections import Counter

sys.path.append('../aenets')


def run_on_model(dataset, nc, ndim, cluster='k-means'):
    matrix_reduce = np.load('../PC_datas/matrix/{}.npy'.format(dataset))

    ''' bad 
        elif cluster == 'dbscan':
        # DBSCAN
        dbscan = DBSCAN(eps=15, min_samples=30)
        labels = dbscan.fit_predict(matrix_reduce[:, :ndim])
        
        elif cluster == 'aff':
        # 亲和力传播
        affinity_propagation = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15)
        labels = affinity_propagation.fit_predict(matrix_reduce[:, :ndim])
        
        elif cluster == 'm-kmeans':
        mkmeans = MiniBatchKMeans(n_clusters=nc, batch_size=150, max_iter=300)
        # 使用模型拟合数据
        mkmeans.fit(matrix_reduce[:, :ndim])  # data 是你的输入数据
        # 获取聚类结果
        labels = mkmeans.labels_  # 获取每个样本所属的簇
        
        elif cluster == 'ms':
        # 创建 MeanShift 模型
        ms = MeanShift(bandwidth=5, cluster_all=False)
        # 使用模型拟合数据
        ms.fit(matrix_reduce[:, :ndim])  # data 是你的输入数据
        # 获取聚类结果
        labels = ms.labels_  # 获取每个样本所属的簇
        
        elif cluster == 'ms':
        # 创建 MeanShift 模型
        ms = MeanShift(bandwidth=5, cluster_all=False)
        ms.fit(matrix_reduce[:, :ndim])
        labels = ms.labels_
    
    '''

    if cluster == 'k-means':
        # k-means
        kmeans = KMeans(n_clusters=nc, n_init=500).fit(matrix_reduce[:, :ndim])
        labels = kmeans.labels_

    elif cluster == 'agg':
        # 层次聚类（Agglomerative Clustering）
        agg_clustering = AgglomerativeClustering(n_clusters=nc)
        labels = agg_clustering.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'spec':
        # 谱聚类（Spectral Clustering）
        spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        labels = spectral_clustering.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'birch':
        # 平衡迭代
        birch = Birch(n_clusters=nc, threshold=1, branching_factor=50)
        labels = birch.fit_predict(matrix_reduce[:, :ndim])

    elif cluster == 'Gauss':
        gauss = GaussianMixture(n_components=nc, random_state=0)
        labels = gauss.fit_predict(matrix_reduce[:, :ndim])

    
    #elif cluster == 'ensemble':
        # kmeans = KMeans(n_clusters=nc, n_init=500)
        # agg_clustering = AgglomerativeClustering(n_clusters=nc)
        # spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        # birch = Birch(n_clusters=nc, threshold=1, branching_factor=50)
        # ensemble_model = ClusteringEnsemble(models=[kmeans, agg_clustering, spectral_clustering, birch])
        #
        # ensemble_model.fit(matrix_reduce[:, :ndim])
        # labels = ensemble_model.predict(matrix_reduce[:, :ndim])

        # 定义各个聚类器
        #kmeans = KMeans(n_clusters=nc, n_init=500)
        #agg_clustering = AgglomerativeClustering(n_clusters=nc)
        #spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        #birch = Birch(n_clusters=nc, threshold=1, branching_factor=50)

        # 构建集成聚类模型
        #ensemble_model = EnsembleClustering(estimators=[kmeans, agg_clustering, spectral_clustering, birch])

        # 训练集成模型并预测
        #labels = ensemble_model.fit_predict(matrix_reduce[:, :ndim])

    else:
        print('报错也能出问题？再改改！')
        exit(1)

    return labels


def bulk_process():
    # *******************************调参部分*****************************************

    datasets = ['Ramani', '4DN', 'Lee']
    # k-means, dbscan, agg, spec, birch, ensemble
    clusters = ['k-means', 'agg', 'spec', 'birch', 'Gauss']
    result_path = '../PC_datas/results.json'
    notes = ''

    # ********************************************************************************

    for dataset in datasets:
        # 分类数
        if dataset == 'Lee':
            nc = 14
            ndim = 71
        elif dataset == '4DN':
            nc = 5
            ndim = 71
        elif dataset == 'Ramani':
            nc = 4
            ndim = 71
        else:
            assert 0, print('check dataset name!')

        for cluster in clusters:
            cluster_labels = run_on_model(dataset=dataset, nc=nc, ndim=ndim, cluster=cluster)

            y = np.load('../PC_datas/labels/{}_labels.npy'.format(dataset))

            # 计算调整兰德指数和归一化互信息
            ari = adjusted_rand_score(y, cluster_labels)
            nmi = normalized_mutual_info_score(y, cluster_labels)
            hm = homogeneity_score(y, cluster_labels)
            fm = completeness_score(y, cluster_labels)

            with open(result_path, 'r') as f:
                res = json.load(f)

            if dataset not in res:
                res[dataset] = {}
            if cluster not in res[dataset]:
                res[dataset][cluster] = {}

            # 更新结果字典
            res[dataset][cluster]['ARI'] = round(ari, 3)
            res[dataset][cluster]['NMI'] = round(nmi, 3)
            res[dataset][cluster]['HM'] = round(hm, 3)
            res[dataset][cluster]['FM'] = round(fm, 3)
            res[dataset][cluster]['NOTES'] = notes

            with open(result_path, 'w') as outfile:
                json.dump(res, outfile)


if __name__ == '__main__':
    # *******************************调参部分*****************************************

    dataset = 'Lee'
    # k-means, agg, spec, birch, ensemble
    cluster = 'k-means'
    c2 = 'spec'
    result_path = '../PC_datas/results.json'
    is_write = False
    notes = ''

    # ********************************************************************************

    # 分类数
    if dataset == 'Lee':
        nc = 14
        ndim = 71
    elif dataset == '4DN':
        nc = 5
        ndim = 71
    elif dataset == 'Ramani':
        nc = 4
        ndim = 71
    else:
        assert 0, print('check dataset name!')

    cluster_labels = run_on_model(dataset=dataset, nc=nc, ndim=ndim, cluster=cluster)
    cluster_labels2 = run_on_model(dataset=dataset, nc=nc, ndim=ndim, cluster=c2)

    y = np.load('../PC_datas/labels/{}_labels.npy'.format(dataset))

    print('predict :', Counter(list(cluster_labels)))
    print('real :', Counter(list(cluster_labels2)))

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(cluster_labels2, cluster_labels)
    nmi = normalized_mutual_info_score(cluster_labels2, cluster_labels)
    hm = homogeneity_score(cluster_labels2, cluster_labels)
    fm = completeness_score(cluster_labels2, cluster_labels)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("Homogeneity (HM):", hm)
    print("Completeness (FM):", fm)
    print('cluster=', cluster)
    # # 计算调整兰德指数和归一化互信息
    # ari = adjusted_rand_score(y, cluster_labels)
    # nmi = normalized_mutual_info_score(y, cluster_labels)
    # hm = homogeneity_score(y, cluster_labels)
    # fm = completeness_score(y, cluster_labels)
    #
    # print("Adjusted Rand Index (ARI):", ari)
    # print("Normalized Mutual Information (NMI):", nmi)
    # print("Homogeneity (HM):", hm)
    # print("Completeness (FM):", fm)
    # print('cluster=', cluster)

    if is_write:
        with open(result_path, 'r') as f:
            res = json.load(f)

        if dataset not in res:
            res[dataset] = {}
        if cluster not in res[dataset]:
            res[dataset][cluster] = {}

        # 更新结果字典
        res[dataset][cluster]['ARI'] = round(ari, 3)
        res[dataset][cluster]['NMI'] = round(nmi, 3)
        res[dataset][cluster]['HM'] = round(hm, 3)
        res[dataset][cluster]['FM'] = round(fm, 3)
        res[dataset][cluster]['NOTES'] = notes

        print(res)

        with open(result_path, 'w') as outfile:
            json.dump(res, outfile)
