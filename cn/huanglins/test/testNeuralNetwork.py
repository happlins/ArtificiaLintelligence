import numpy as np

"""
:param X 输入数据集，形式为矩阵，每一行代表一个训练样本
:param y 输出数据集，形式为矩阵，每一行代表一个训练样本
:param l0 神经网络第1层，即网络输入层
:param l1 神经网络第2层，常称作隐藏层
:param syn0 第一层权值，突触0，连接l0层与l1层。

"""

# sigmoid function
def nonlin(x,deriv=False):
    """
    把线性，转变为非线性,可以把线性函数转变成0到1之间的值
    :param x:
    :param deriv: 是否求导
    :return:
    """
    if (deriv == True):
        return x*(1-x)
    # 下面式子相当于，1/1+e^-1 ,可以一个线性变为非线性的,称作 “sigmoid” 的函数
    # 映射到0-1之间
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

print("X.shape",X.shape)
# output dataset
y = np.array([[0,0,1,1]]).T
print("y.shape:",y.shape)

# 发送随机数字进行计算
# 确定性的（只是一个好的做法）,生成同一随机数
np.random.seed(1)

# 用均值0随机初始化权重
syn0 = 2*np.random.random((3,1)) - 1

print("syn0:" , syn0)

for iter in range(10000):
    # 前向传播
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # 我们错过了多少？
    l1_error = y - l1

    # 乘以我们在l1中的值与sigmoid的斜率相差多少
    l1_delta = l1_error * nonlin(l1,True)

    # 更新权重
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)