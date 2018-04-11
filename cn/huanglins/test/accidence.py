from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 参数概要
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean",mean) # 平均值
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev",stddev) # 标准差
        tf.summary.scalar("max",tf.reduce_max(var)) # 最大值
        tf.summary.scalar("min",tf.reduce_min(var)) # 最小值
        tf.summary.histogram("histogram",var) # 直方图


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_batch = mnist.train.num_examples // 100
# 定义命令空间
with tf.name_scope("input"):
    # 放置占位符，用于在计算时接收输入值
    x = tf.placeholder("float", [None, 784],name="x-input")
    # 为了进行训练，需要把正确值一并传入网络
    y_ = tf.placeholder("float", [None, 10],name="y_-input")


with tf.name_scope("layer"):
    # Variable为可以改变的量
    # 创建两个变量，分别用来存放权重值W和偏置值b
    # 第一参数为图片的像素，第二参数为每个像素对应的数字的权重（就是乘以某个数）
    with tf.name_scope("wights"):
        W = tf.Variable(tf.zeros([784, 10]),name="W")
        variable_summaries(W)
    with tf.name_scope("biases"):
        # 偏置量，相当于误差
        b = tf.Variable(tf.zeros([10]),name="b")
        variable_summaries(b)
    # 使用Tensorflow提供的回归模型softmax，y代表输出
    # tf.matmul(x,W)表示x和W相乘
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope("softmax"):
        y = tf.nn.softmax(wx_plus_b)



# 计算交叉熵,说明白就是损失函数，用来计算误差
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
with tf.name_scope("cross_entropy"):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar("cross_entropy",cross_entropy)
# 使用梯度下降算法以0.01的学习率最小化交叉墒
with tf.name_scope("train_setp"):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化之前创建的变量的操作
init = tf.global_variables_initializer()


# 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
        # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy",accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

# 启动初始化
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("logs/",sess.graph)

# 开始训练模型，循环1000次，每次都会随机抓取训练数据中的100条数据，然后作为参数替换之前的占位符来运行train_step
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _,summary = sess.run([train_step,merged], feed_dict={x: batch_xs, y_: batch_ys})
    writer.add_summary(summary,i)

# 在session中启动accuracy，输入是MNIST中的测试集
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
