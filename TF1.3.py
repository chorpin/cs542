import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy

state = tf.Variable(0.0)
add_op = tf.assign(state, state+tf.constant(1.0))
#assign的结果也是float32
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #记得initializaer后面加括号
    print ('init state:', sess.run(state), 'type:', type(sess.run(state)))
    for _ in range(3):
        sess.run(add_op)
        print(sess.run(state))
        print('sess_op:', sess.run(add_op), 'sess_op_type:', type(sess.run(add_op)))

    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.add(tf.multiply(input1, input2), 1)

    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input1:[7.0], input2:[2.0]}))

