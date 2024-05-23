import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score

dataset = 'Ramani'
sdir1 = 'diag8'
extra1 = 'm20_o6'
train_epochs1 = 500
prct1 = 30
# k-means, agg, spec, ensemble
cluster1 = 'agg2'
a1 = 'code'

sdir2 = 'diag8'
extra2 = 'm20_o6_test'
train_epochs2 = 500
prct2 = 30
# k-means, agg, spec, ensemble
cluster2 = 'agg2'
a2 = 'code'

cluster_dir1 = '../PC_datas/predict_labels/{}'.format(dataset)
cluster_file1 = '{}_{}_{}_{}_{}_{}.npy'.format(sdir1, extra1, train_epochs1, prct1, cluster1, a1)
cluster_path1 = os.path.join(cluster_dir1, cluster_file1)

cluster_dir2 = '../PC_datas/predict_labels/{}'.format(dataset)
cluster_file2 = '{}_{}_{}_{}_{}_{}.npy'.format(sdir2, extra2, train_epochs2, prct2, cluster2, a2)
cluster_path2 = os.path.join(cluster_dir2, cluster_file2)

pl1 = np.load(cluster_path1)
pl2 = np.load(cluster_path2)

# 计算调整兰德指数和归一化互信息
ari = adjusted_rand_score(pl1, pl2)
nmi = normalized_mutual_info_score(pl1, pl2)
hm = homogeneity_score(pl1, pl2)
fm = completeness_score(pl1, pl2)

print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)
print("Homogeneity (HM):", hm)
print("Completeness (FM):", fm)

print(cluster_path1)
print(cluster_path2)
