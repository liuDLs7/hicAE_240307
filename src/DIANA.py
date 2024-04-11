# import numpy as np
# from scipy.spatial.distance import cdist
#
# class DIANA:
#     def __init__(self, nclusters: int, random_state=None):
#         self.nclusters = nclusters
#         self.random_state = random_state
#
#     def split_one_cluster(self, cluster, dis_matrix):
#         np.random.seed(self.random_state)
#         temp_dis_matrix = dis_matrix[cluster][:, cluster]
#         max_dis_index = np.argmax(np.mean(temp_dis_matrix, axis=1))
#         id_split = cluster[max_dis_index]
#
#         split_cluster = [id_split]
#         last_cluster = [c for c in cluster if c != id_split]
#
#         while True:
#             flag_split = False
#             for i in range(len(last_cluster) - 1, -1, -1):
#                 p = last_cluster[i]
#                 dis_p_split = dis_matrix[p, split_cluster]
#                 dis_p_last = dis_matrix[p, last_cluster]
#
#                 if np.mean(dis_p_split) <= np.mean(dis_p_last):
#                     split_cluster.append(p)
#                     last_cluster.pop(i)
#                     flag_split = True
#             if not flag_split:
#                 break
#
#         return split_cluster, last_cluster
#
#     def get_max_separation_cluster(self, clusters, dis_matrix):
#         np.random.seed(self.random_state)
#         dgree_separation = [np.max(dis_matrix[cluster][:, cluster]) for cluster in clusters]
#         return np.argmax(dgree_separation)
#
#     def fit_predict(self, data):
#         np.random.seed(self.random_state)
#         N, D = data.shape
#         tile_x = np.tile(data[:, np.newaxis, :], (1, N, 1))
#         tile_y = np.tile(data[np.newaxis, :, :], (N, 1, 1))
#         dis_matrix = np.linalg.norm(tile_x - tile_y, axis=-1)
#
#         clusters = [list(range(N))]
#         K = self.nclusters
#
#         while len(clusters) < K:
#             index_sel = self.get_max_separation_cluster(clusters, dis_matrix)
#             c_1, c_2 = self.split_one_cluster(clusters[index_sel], dis_matrix)
#
#             clusters.pop(index_sel)
#             clusters.extend([c_1, c_2])
#
#         cluster_labels = np.zeros(N)
#         for i, cluster in enumerate(clusters):
#             cluster_labels[cluster] = i
#
#         return cluster_labels
#
#
#     # def fit_predict(self, data):
#     #     np.random.seed(self.random_state)
#     #     N, D = data.shape
#     #
#     #     dis_matrix = cdist(data, data)  # 使用cdist计算距离矩阵
#     #
#     #     clusters = [list(range(N))]
#     #     K = self.nclusters
#     #
#     #     while len(clusters) < K:
#     #         index_sel = self.get_max_separation_cluster(clusters, dis_matrix)
#     #         c_1, c_2 = self.split_one_cluster(clusters[index_sel], dis_matrix)
#     #
#     #         clusters.pop(index_sel)
#     #         clusters.extend([c_1, c_2])
#     #
#     #     cluster_labels = np.zeros(N)
#     #     for i, cluster in enumerate(clusters):
#     #         cluster_labels[cluster] = i
#     #
#     #     return cluster_labels

import numpy as np

class DIANA:
    def __init__(self, nclusters: int, random_state=None):
        self.nclusters = nclusters
        self.random_state = random_state

    def euclidean_distance(self, x, y):
        return np.linalg.norm(x - y)

    def split_one_cluster(self, cluster, dis_matrix):
        np.random.seed(self.random_state)
        temp_dis_matrix = dis_matrix[cluster][:, cluster]
        max_dis_index = np.argmax(np.mean(temp_dis_matrix, axis=1))
        id_split = cluster[max_dis_index]

        split_cluster = [id_split]
        last_cluster = [c for c in cluster if c != id_split]

        while True:
            flag_split = False
            for i in range(len(last_cluster) - 1, -1, -1):
                p = last_cluster[i]
                dis_p_split = dis_matrix[p, split_cluster]
                dis_p_last = dis_matrix[p, last_cluster]

                if np.mean(dis_p_split) <= np.mean(dis_p_last):
                    split_cluster.append(p)
                    last_cluster.pop(i)
                    flag_split = True
            if not flag_split:
                break

        return split_cluster, last_cluster

    def get_max_separation_cluster(self, clusters, dis_matrix):
        np.random.seed(self.random_state)
        dgree_separation = [np.max(dis_matrix[cluster][:, cluster]) for cluster in clusters]
        return np.argmax(dgree_separation)

    def fit_predict(self, data):
        np.random.seed(self.random_state)
        N, D = data.shape
        tile_x = np.tile(data[:, np.newaxis, :], (1, N, 1))
        tile_y = np.tile(data[np.newaxis, :, :], (N, 1, 1))
        dis_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                dis_matrix[i, j] = self.euclidean_distance(data[i], data[j])

        clusters = [list(range(N))]
        K = self.nclusters

        while len(clusters) < K:
            index_sel = self.get_max_separation_cluster(clusters, dis_matrix)
            c_1, c_2 = self.split_one_cluster(clusters[index_sel], dis_matrix)

            clusters.pop(index_sel)
            clusters.extend([c_1, c_2])

        cluster_labels = np.zeros(N)
        for i, cluster in enumerate(clusters):
            cluster_labels[cluster] = i

        return cluster_labels
