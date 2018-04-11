# -- coding:utf-8 --
import numpy as np

class Perceptron(object):
    """
    :param eta: 学习率
    :param n_iter:权重向量的训练次数
    :param w_:神经分叉重向量
    :param errors_: 用于记录神经元判断出错次数
    """
    def __init__(self,eta = 0.01,n_iter=10):
        """
        :param eta: 学习率
        :param n_iter: 权重向量的训练次数
        """
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self,X,y):
        """
        输入训练数据，培训神经元，X输入样本向量，y对应样本分类
        :param X:shape[n_samples 到底有多少个输入样本,n_features 神经元的分叉，意思就是接受多少个输入]
        举例:X（向量）:[[1,2,3],[4,5,6]]
        n_samples:为向量的个数:2
        n_features:为向量中的元素的个数:3
        :param y:[1,-1],对应两个向量的结果
        :return:
        """

        """
        初始化权重向量为0
        加一是因为前面算法提到的w0,也就是步调函数(激活函数)的阈值
        """
        self.w_ = np.zeros(1 + X.shape(1))
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update;

                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
        pass

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
        pass

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0 ,1,-1)

    pass

import pandas as pd
file = "iris.data.csv"
df = pd.read_csv(file,header=None)
print(df.head(10))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "SimHei"

y = df.loc[0:100,4].values
y = np.where(y == "Iris-setosa",-1,1)

X = df.iloc[0:100,[0,2]].values

# plt.scatter(X[:50,0],X[:50,1],color="red",marker="o",label="setosa")
# plt.scatter(X[50:,0],X[50:,1],color="blue",marker="x",label="versicolor")
# plt.xlabel("花瓣长度")
# plt.ylabel("花径长度")
# plt.legend(loc="upper left")
# plt.show()

ppn = Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_) + 1),ppn.errors_,marker="o")
plt.xlabel("Epochs")
plt.ylabel("错误分类次数")
plt.show()