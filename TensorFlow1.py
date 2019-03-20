import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, 1)
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(state))
    for x in range(3):
        sess.run(update)
        print (sess.run(state))
    print (type(one))
print ('********1.2********', '\n')
h_sum = tf.Variable(0.0, dtype = tf.float32)
h_vec = tf.constant([1.0, 2.0, 3.0, 4.0])
h_add = tf.placeholder(tf.float32)
h_new = tf.add(h_sum, h_add)
update = tf.assign(h_sum, h_new)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('s_sum = ', sess.run(h_sum))
    print('vec = ', sess.run(h_vec))
    for _ in range(4):
        sess.run(update, feed_dict={h_add: sess.run(h_vec[_])})
        print('h_sun = ', sess.run(h_sum))
    print('the mean is:', sess.run(sess.run(h_sum)/tf.constant(4.0)))
    print(type(h_sum))
    # The following result is <class 'numpy.float32'>
    print(type(sess.run(h_sum)))