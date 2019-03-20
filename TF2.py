import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

print('training data shape:', mnist.train.images.shape)
# (55000, 28x28=784)
print('training label shape:', mnist.train.labels.shape)

print('#####构建网络######')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32,[None, 10])

# FC1
W_fc1 = weight_variable([784, 1024])
# b_fc1的size
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(X_, W_fc1) + b_fc1)

# FC2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pre = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 损失函数： 交叉熵函数 cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pre))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1))
#tf.cast is used to change the data type
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
X_batch, y_batch = mnist.train.next_batch(batch_size = 100)
# for i in range(5000):
#     X_batch, y_batch = mnist.train.next_batch(batch_size = 100)
#     train_step.run(feed_dict={X_: X_batch, y_: y_batch})
#     if (i + 1) % 200 == 0:
#         train_accuracy = accuracy.eval(feed_dict={X_: mnist.train.images, y_: mnist.train.labels})
#         print('step %d, training acc %g'% (i + 1, train_accuracy))
#     if (i + 1) % 1000 == 0:
#         test_accuracy = accuracy.eval(feed_dict={X_: mnist.test.images, y_: mnist.test.labels})
#         print('= ' * 10, 'step %d, testing acc %g' % (i + 1, test_accuracy))

print('mnist.train:', mnist.train)
print('\n')
# type <class 'numpy.ndarray'> 多维数组
print('X_batch:', X_batch, 'type', type(X_batch), '\n', len(X_batch), '*',len(X_batch[0]))
