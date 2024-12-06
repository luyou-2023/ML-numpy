import numpy as np
import matplotlib.pyplot as plt

'''
1. 模型公式
2. 训练数据，随机参数预测值
3. 损失函数求预测值与真实值的差异
4. 对损失函数求导，求出参数变为与损失值的关系
5. 根据导数更新参数，使得损失值越来越小，预测值接近真实值，次数的参数逼近最优解
'''
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
            多元线性回归模型的损失函数对参数的导数 
            假设多元线性回归模型的表达式为：y=Xθ 
            其中：y 是目标值的向量，形状为(m,1)，m 是样本数量。
            X 是设计矩阵 形状为 (m,n)，每一行是一个样本的特征向量，n 是特征数量。
            θ 是权重参数向量 形状为 (n,1)。
            X@θ 是模型的预测值（形状为 (m,1)）。
            损失函数: 均方误差（MSE）L(θ)= 1/2m |Xθ−y| Xθ−y 是误差（残差）∥⋅∥ 2是向量的平方和。
            为了优化模型参数 θ，我们对损失函数 𝐿(𝜃) 对 θ 求导。对矩阵运算求导的结果为：∂L(θ)/∂θ = 1/m xT(Xθ−y) 其中∂θ是导数、梯度 求解出来更新参数，∂θ的变化对参数的影响，使得损失值最小
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
