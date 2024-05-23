import numpy as np

t = ''
b ='abcd'
a=[1,2,3,4,5]
if 'code' in 'diag8_m20_o6_500_30_agg.npy':
    print('yes')
else:
    print('no')



exit(0)

# 获取每个对角线中非零值的个数
nonzero_counts = [np.count_nonzero(np.diagonal(matrix, offset=i)) for i in range(0, matrix.shape[1])]
# 获取每个对角线上元素的和
diagonal_sums = [np.sum(np.diagonal(matrix, offset=i)) for i in range(0, matrix.shape[1])]

print("每个对角线中非零值的个数:", nonzero_counts)
print("每个对角线上元素的和:", diagonal_sums)

# 创建两个示例数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6, 7])

# 将两个数组相加
result = array1 + array2

print("相加的结果:", result)