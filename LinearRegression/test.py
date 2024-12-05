import numpy as np

# 一维数组
a = np.array([1, 2, 3])
print(a)
print(a.shape)  # (3,)

# 在第 0 轴增加一个维度
b = a[np.newaxis, :]
print(b)
print(b.shape)  # (1, 3)

# 在第 1 轴增加一个维度
c = a[:, np.newaxis]
print(c)
print(c.shape)  # (3, 1)
