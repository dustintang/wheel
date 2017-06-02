#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:50:55 2017

@author: dustin
"""

import tensorflow as tf
import sys
sys.path.append("/Users/dustin/Documents/study/Github/Python/wheel/ml")
import input_data
#导入数据
mnist = input_data.read_data_sets('data/', one_hot=True) #从网站获取mnist数据
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST ready")
#参数设置
n_input  = 784
n_output = 10
weights  = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
    }
biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
#模型构建
def conv_basic(_input, _w, _b, _keepratio): #数据，w，b两个参数，和dropout的保留率
        # INPUT 输入转化为tf的四维数据格式
        _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
        # CONV LAYER 1 第一个卷积层
        _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        #_mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
        #_conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001) #BN批规范化，有助于模型优化
        _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1'])) #卷积层后的激活函数
        _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #池化层
        _pool_dr1 = tf.nn.dropout(_pool1, _keepratio) #池化操作后的dropout操作
        # CONV LAYER 2 第二个卷积层
        _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        #_mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
        #_conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
        _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
        _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
        # VECTORIZE 向量化
        _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
        # FULLY CONNECTED LAYER 1 全连接层，得到一定维度的向量数据，并dropout
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # FULLY CONNECTED LAYER 2 全连接层，分类中各类概率
        _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        # RETURN 返回值
        out = { 'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
            'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
        }
        return out
print ("CNN READY")

#待传入数据的占位
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS 传入参数到模型中，并初始化
_pred = conv_basic(x, weights, biases, keepratio)['out'] #将参数传入模型中
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels= y)) #loss
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) #优化操作
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) 
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))  #精度指标
init = tf.global_variables_initializer() #初始化
    
# SAVER 保存
save_step = 1
saver = tf.train.Saver(max_to_keep=3) #最多保存3个版本

do_train = 1 #设置是否执行保存和展示，还是载入已保存
sess = tf.Session()
sess.run(init) #执行所有变量的初始化

training_epochs = 15 #所有样本训练15次
batch_size      = 50 #每次输入的样本量，即batch大小为16
display_step    = 1  #展示数据的频次，每个epoch后展示一次
if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size) #由样本量除以batch得到batch的次数
        total_batch = 10
        # Loop over all batches 所有batches进行循环迭代
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #得到next batch的数据和标签
            # 用batch数据训练模型。 将batch数据和dropout的留存率作为参数传入optm中涉及到的x，y，keepratio三个占位参数中
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
            # Compute average loss 计算平均loss值
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch
        # Display logs per epoch step 展示学习效果
        if epoch % display_step == 0: 
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
            print (" Training accuracy: %.3f" % (train_acc))
            test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
            print (" Test accuracy: %.3f" % (test_acc))          
        # Save Net 保存模型
        if epoch % save_step == 0:
            saver.save(sess, "save_cnn/cnn_mnist_basic.ckpt-" + str(epoch))
    print ("OPTIMIZATION FINISHED")
    
#导入已保存的计算后的模型参数，并防止重复计算
if do_train == 0:
    epoch = training_epochs-1
    saver.restore(sess, "save/nets/cnn_mnist_basic.ckpt-" + str(epoch))   
    test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
    print (" TEST ACCURACY: %.3f" % (test_acc))