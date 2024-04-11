from sklearn.cluster import \
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch, MiniBatchKMeans, MeanShift
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
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
import ClusterEnsembles as CE
import markov_clustering as mc
import networkx as nx
import random
from DIANA import DIANA
from sklearn.manifold import TSNE

sys.path.append('../aenets')

# 计算余弦相似性矩阵
def cosine_similarity_matrix(features):
    # 计算特征向量的范数
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    # 归一化特征向量
    normalized_features = features / norm
    # 计算相似性矩阵
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    return similarity_matrix


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

    elif cluster == 'mc':
        similarity_matrix = cosine_similarity_matrix(matrix_reduce[:, :ndim])
        result = mc.run_mcl(similarity_matrix, inflation=2.5)           # run MCL with default parameters
        clusters = mc.get_clusters(result)    # get clusters
        print(clusters)
        exit(0)

    elif cluster == 'diana':
        diana = DIANA(nclusters=nc, random_state=1024)
        labels = diana.fit_predict(TSNE(learning_rate=750, random_state=1024).fit_transform(matrix_reduce[:, :ndim]))

    elif cluster == 'ensemble':
        kmeans = KMeans(n_clusters=nc, n_init=500).fit(matrix_reduce[:, :ndim])
        l1 = list(kmeans.labels_)
        agg_clustering = AgglomerativeClustering(n_clusters=nc)
        l2 = list(agg_clustering.fit_predict(matrix_reduce[:, :ndim]))
        spectral_clustering = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_init=100)
        l3 = list(spectral_clustering.fit_predict(matrix_reduce[:, :ndim]))
        labels = np.array([l1, l2, l3])
        labels = CE.cluster_ensembles(labels, solver='mcla')

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
    clusters = ['k-means', 'agg', 'spec', 'birch']
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
    cluster = 'diana'
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

    y = np.load('../PC_datas/labels/{}_labels.npy'.format(dataset))

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
    print('cluster=', cluster)

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
