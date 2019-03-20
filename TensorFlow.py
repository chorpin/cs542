#coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy
x_data = numpy.float32(numpy.random.rand(2, 100))
#t1 = numpy.array([[1, 3, 5], [2, 4, 6]])
y_data = numpy.dot([1.0, 2.0], x_data) + 0.3
#print('t2:', t2)
b = tf.Variable(tf.zeros([1]))
## b is just a number
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
#print('W:', W)


y = tf.matmul(W, x_data) + b
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(y, '\n', sess.run(y))
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

###################

init = tf.global_variables_initializer()
config = tf.ConfigProto()
#
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))
input1 = tf.constant(2.0)
input2 = tf.constant(3.0)
input3 = tf.constant(5.0)

intermd = tf.add(input1, input2)
mul = tf.multiply(input2, input3)

with tf.Session() as sess: # 不用with也可以，测试了
    result = sess.run([mul, intermd])  # 一次执行多个op
    print (result)
    print (type(result))
    print (type(result[0]) )
