import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
# print('validation:', mnist.validation.labels.shape)
# print('validation picture:', mnist.validation.images.shape)
# print('test:', mnist.test.labels.shape)
# print('train:', mnist.train.labels.shape)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# 定义卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding ='SAME')

# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

###预留

import os
train_list = os.listdir('C:\\Users\\王嘉炜\\Desktop\\machine learning projects\\training')
#先是T7 后是lambda，各250个
print(train_list)
train_pic = np.zeros([500, 262144])

tl_a = np.ones([250, 1])
tl_b = np.zeros([250, 1])
train_labelc = np.vstack((tl_a, tl_b))
train_labeld = np.vstack((tl_b, tl_a))
train_label = np.hstack((train_labelc, train_labeld))
#train_label = tf.Variable(train_label, tf.float32)
i = 0
for x in train_list:
    rgb = np.loadtxt("C:\\Users\\王嘉炜\\Desktop\\machine learning projects\\training\\%s"%train_list[i])
    gs = rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
    train_pic[i, :] = gs
    i = i+1
print('train_pic_size:', train_pic.shape)
#train_pic = tf.Variable(train_pic, tf.float32)



test_list = os.listdir('C:\\Users\\王嘉炜\\Desktop\\machine learning projects\\testing')
test_pic = np.zeros([100, 262144])

l_a = np.ones([50, 1])
l_b = np.zeros([50, 1])
test_labelc = np.vstack((l_a, l_b))
test_labeld = np.vstack((l_b, l_a))
test_label = np.hstack((test_labelc, test_labeld))
#test_label = tf.Variable(test_label, tf.float32)

j = 0
for tx in test_list:
    trgb = np.loadtxt("C:\\Users\\王嘉炜\\Desktop\\machine learning projects\\testing\\%s"%test_list[j])
    tgs = trgb[:, 0] * 0.299 + trgb[:, 1] * 0.587 + trgb[:, 2] * 0.114
    test_pic[j, :] = tgs
    j = j+1
print('test_pic_size:', test_pic.shape)
#test_pic = tf.Variable(test_pic, tf.float32)
###


#X_ = tf.placeholder(tf.float32, [None, 784])# 改为512x512
X_ = tf.placeholder(tf.float32, [None, 262144])
#y_ = tf.placeholder(tf.float32, [None, 10])# 改为2
y_ = tf.placeholder(tf.float32, [None, 2])

X = tf.reshape(X_, [-1, 512, 512, 1])# 改为512x512
#把X转化为卷积所需要的形式
# print(X),
# the result: Tensor("Reshape:0", shape=(?, 28, 28, 1), dtype=float32)

# W就是filter 32个 size是5x5
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

# 第一个pooling层
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积： 5x5x32 卷积核64个 [5,5,32,64], h_conv2.shape=[-1,14,14,64]

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个pooling层，[-1, 14, 14, 64] -> [-1, 7*7*64], 就是说把这个东西展开
#然后塞到full connected里

h_pool2 = max_pool_2x2(h_conv2)
# flatten层
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 128*128*64])#两次pool之后 变为四分之一

# fc1
#W_fc1 = weight_variable([7*7*64, 1024])
W_fc1 = weight_variable([128*128*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#防止过度拟合 drop 随机的一些元素变成0 剩下的变成原值的 1/keep_prob
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#W_fc2 = weight_variable([1024, 10])
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#以下是训练和评估
cross_entropy = -tf.reduce_sum(y_ *tf.log(y_conv))
# 注意： 信息熵直接是和最终结果比较的 1e-4 = 0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_acc_sum = tf.Variable(0.0)
batch_acc = tf.placeholder(tf.float32)
new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
update = tf.assign(test_acc_sum, new_test_acc_sum)


sess.run(tf.global_variables_initializer())
# for i in range(5000):
#     X_batch, y_batch = mnist.train.next_batch(batch_size=50)
#     if i % 500 == 0:
#         train_accuracy = accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0})
#         print(i, 'accuracy:', train_accuracy)
#         print('X_batch:', X_batch, 'shape:', X_batch.shape)
#     train_step.run(feed_dict = {X_: X_batch, y_: y_batch, keep_prob: 0.5})
#
# print('X_batch:', X_batch, 'shape:', X_batch.shape)
    #删除这条语句 运行很快 好像是跳过了 正确率很低0.2-0.3
    #将prob设置为 1 结果还不错 0.18（0）-0.9（500）-0.96（1000）-1（1500）-0.96（2000）-0.98（2500）-0.98（3000）-0.98（3500）-
###------------------------------------------------------------------------------------######
X_batch, y_batch = train_pic, train_label
for i in range(5000):
    if i % 500 == 0:
        train_accuracy = accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0})
        print(i, 'accuracy:', train_accuracy)
        print('X_batch:', X_batch, 'shape:', X_batch.shape)
    train_step.run(feed_dict = {X_: X_batch, y_: y_batch, keep_prob: 0.5})

#print('X_batch:', X_batch, 'shape:', X_batch.shape)
###------------------------------------------------------------------------------------#######

    # 为什么要重置 prob的值
#全部训练完了，再做测试，batch_size = 100
# for i in range(100):
#     X_batch, y_batch = mnist.test.next_batch(batch_size = 100)
#     test_acc = accuracy.eval(feed_dict = {X_: X_batch, y_: y_batch, keep_prob: 1.0})
#     update.eval(feed_dict = {batch_acc: test_acc})
#     if (i+1)%20 == 0:
#         print('testing step', i+1, 'test_acc_sum', test_acc_sum.eval())
# print('test_accuracy:', test_acc_sum.eval()/100.0)

###----------------------------------------------------------------------------##########







for i in range(100):
    X_batch, y_batch = test_pic, test_label
    test_acc = accuracy.eval(feed_dict = {X_: X_batch, y_: y_batch, keep_prob: 1.0})
    update.eval(feed_dict = {batch_acc: test_acc})
    if (i+1)%20 == 0:
        print('testing step', i+1, 'test_acc_sum', test_acc_sum.eval())
print('test_accuracy:', test_acc_sum.eval()/100.0)




# img1 = mnist.train.images[1]
# label1 = mnist.train.labels[1]
# print(label1)
# print('1. img_data shape =', img1.shape)
# img1.shape = [28, 28]
# import matplotlib.pyplot as plt
# print('2. img_data shape =', img1.shape)
#print(img1) 28x28 的二维矩阵
# img2= np.random.random((300,300))
# plt.subplot(4,8,1)
# plt.imshow(img1, cmap = 'gray')
# plt.axis('off')
# plt.subplot(4,8,3)
# plt.imshow(img1, cmap = 'hot')
# plt.axis('off')
# plt.subplot(4,8,5)
# plt.imshow(img1)
# plt.axis('off')
# plt.show()
# X_img = img1.reshape([-1, 784])
# y_img = mnist.train.labels[1].reshape([-1, 10])
# result = h_conv1.eval(feed_dict={X_: X_img, y_: y_img, keep_prob: 1.0})
# print(result.shape)
# print(type(result))





#plot the example
# greyScale=a[:,0]*0.299 + a[:,1]*0.587 + a[:,2]*0.114
# greyScale = greyScale.reshape((512, 512))
# print('greyScale:',greyScale,greyScale.shape)
# import matplotlib.pyplot as plt
# plt.imshow(greyScale, cmap = 'gray')
# plt.axis('off')
# plt.show()

