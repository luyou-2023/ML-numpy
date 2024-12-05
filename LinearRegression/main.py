import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.0001, thr=1e-6):
        self.lr = lr
        self.thr = thr

    def predict(self, X):
        paddedX = np.ones([X.shape[0], X.shape[1] + 1])
        paddedX[:, :-1] = X
        return paddedX @ self.theta  # 计算预测值

    def fit(self, X, y):
        '''

        Args:
            X:
            y:
                    X = [
                        [1.2],
                        [3.4],
                        [5.6],
                        ...,
                        [9.8]
                    ]
        Returns:

        '''
        # 构造theta
        '''
        随机
        theta = [
            [0.3],  # 权重
            [0.5]   # 偏置
        ]
        '''
        self.theta = np.random.randn(X.shape[-1] + 1)[:, np.newaxis]
        # 构造扩展的X
        '''
        paddedX = [
            [1.2, 1],
            [3.4, 1],
            [5.6, 1],
            ...,
            [9.8, 1]
        ]
        '''
        paddedX = np.ones([X.shape[0], X.shape[1] + 1])
        paddedX[:, :-1] = X
        # 开始进行梯度下降
        epoch = 0
        while True:
            # 计算loss
            # paddedX @ self.theta  y=w⋅x+b y=XQ
            '''
            y_pred = paddedX @ theta = [
                [25.6],
                [72.4],
                [119.2],
                ...,
                [200.1]
            ]
            paddedX 是扩展后的特征矩阵，每行代表一个样本，最后一列为常数 1，专门用于计算偏置项
            self.theta 是参数向量，包含权重w 和偏置b
            特征部分: 每一行的特征值与对应的权重相乘 
            偏置部分: 常数 1 和偏置b 相乘，直接加到结果中
            样本矩阵 paddedX 的一行是 [x1,x2,1] 参数向量 self.theta 是[w1,w2,b]
            矩阵乘法：paddedX @ self.theta =  [x1,x2,1] @ [w1,w2,b] = w1*x1 + w2*x2 + b 
            @ 点积就是w1*x1 + w2*x2 + b 
            '''
            loss = 0.5 * np.mean((paddedX @ self.theta - y) ** 2)
            print("第{}次训练，loss大小为{}".format(epoch, loss))
            # 计算梯度
            '''
            grad = paddedX.T @ (paddedX @ self.theta - y)
            计算损失函数对 θ 的导数
            '''
            grad = paddedX.T @ (paddedX @ self.theta - y)
            # 是否收敛
            if abs(np.sum(grad)) < self.thr:
                break
            # 梯度下降
            self.theta -= self.lr * grad
            epoch += 1


if __name__ == "__main__":
    data_size = 100
    # 如 x = [1.2, 3.4, 5.6, ..., 9.8]
    x = np.random.uniform(low=1.0, high=10.0, size=data_size)
    # 因变量（目标值），通过 y = x * 20 + 10 + noise 生成，其中 noise 是正态分布的噪声 y = [35.6, 78.3, 122.1, ..., 210.4]
    y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)
    lr = LinearRegression()
    # 将 x 从一维向量（形状 (100,)）变为二维列向量（形状 (100, 1)）。
    '''
    x = [1.2, 3.4, 5.6, ..., 9.8]
    X = [
        [1.2],
        [3.4],
        [5.6],
        ...,
        [9.8]
    ]
    '''
    lr.fit(x[:, np.newaxis], y[:, np.newaxis])
    plt.scatter(x, y, marker='.')
    plt.plot([1, 10], [lr.theta[0] * 1 + lr.theta[1], lr.theta[0] * 10 + lr.theta[1]], "m-")
    plt.show()


'''
# 假设 paddedX 是：
# [[x1_1, x1_2, 1],
#  [x2_1, x2_2, 1],
#  ...]
# self.theta 是：
# [[w1],
#  [w2],
#  [b]]

# 结果预测值 y_hat:
y_hat = paddedX @ self.theta

# y_hat 的每一行结果表示每个样本的预测值：
# [[x1_1 * w1 + x1_2 * w2 + b],
#  [x2_1 * w1 + x2_2 * w2 + b],
#  ...]
'''
