#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:45:02 2017

@author: dustin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
# settings  超参数设置
#设置：学习率10的负4次方
LEARNING_RATE = 1e-4
#本地设置到2万次迭代，可以达到99的准确率。dropout为50%，每次神经元有5成drop掉。每次样本量在50个。如果总共有100个样本则要两次迭代才能完成1个epoch。
TRAINING_ITERATIONS = 2500        
DROPOUT = 0.5
BATCH_SIZE = 50
#测试集大小，设为0则是训练全部数据
VALIDATION_SIZE = 2000
#最重要输出的类别数量（分类）
IMAGE_TO_DISPLAY = 10
#读取数据文件
data = pd.read_csv('/Users/dustin/Documents/study/Database/test/ml/tensorflow/train.csv')



#1预处理——————
images = data.iloc[:,1:].values  #只有灰度数据，即只有一个通道，正常rgb是3个通道，最大255
images = images.astype(np.float) #转化int64为float64
images = np.multiply(images, 1.0 / 255.0) #归一化操作
image_size = images.shape[1] #图片像素点
#转换成图片的格式，先设置长宽，这里图片都是方的，所以sqrt
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)#设置图片宽和高
#将像素格式转换成图片，用来展示图片
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
# output image  输出图片   
display(images[0])
#values将dataframe格式变成int64，ravel将42000*1变成到1个list
labels_flat = data[[0]].values.ravel()
#计算共有多少类
labels_count = np.unique(labels_flat).shape[0]
#用独热编码将离散特征的取值扩展到欧氏空间
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#one hot编码
labels = dense_to_one_hot(labels_flat, labels_count) #ndarray,type为int64
labels = labels.astype(np.uint8) #ndarray，type为uint8
#分数据集为训练集和验证集（测试集）
validation_images = images[:VALIDATION_SIZE] #验证集的特征和标签
validation_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:] #训练集的特征和标签
train_labels = labels[VALIDATION_SIZE:]



#2模型构建————————
#  权重W矩阵的初始化，高斯分布
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) #tensorflow中所有变量都要初始化操作，只支持这种格式
# b参数，进行常量的初始化操作
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) #常数值
    return tf.Variable(initial)
#卷积操作，传入x与权重项w，常求内积，strides头尾通常为1，中间两个分别代表如何滑动，沿着长和宽的移动。padding是填充，将图片边缘位置也可以滑动到，在原始图像周围填0.
def conv2d(x, W): #w可以是 weight_variable([5, 5, 1, 32])
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #第一个1是batch，最后一个1是channel通道／深度。 padding只有same和valid两种模式，same会完全自动划，凑边缘，valid只做普通边缘项
#池化操作，pooling操作常见的有max pooling，也是一次特征的提取，没有权重参数，只把窗口内最大值提取。指定扩大的窗口和滑动步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 输入数据
# images 。其中placeholder就是先定义结构，none是先不指定有多大的样本，更具后去传入的每个batch的大小再定，先占坑。
x = tf.placeholder('float', shape=[None, image_size])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])

#第一个卷积层，5*5代表窗口大小，窗口内进行一次特征提取。1和32是通道，是w参数连接前后两层的设置，1是连接的前一层，只有一个灰度通道，就是1，32是想要得到的特征图数量，就是用32种w参数得到32种特征图
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32]) #32个特征图对应的32个b
# (40000,784) => (40000,28,28,1)
image = tf.reshape(x, [-1,image_width , image_height,1])
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1) #进行卷积操作，连上激活函数relu函数，就是小于0为0，大于0为y=x
h_pool1 = max_pool_2x2(h_conv1) #卷积层结束后一般不改变像素大小，池化层是降低维度，这里是28*28变成14*14的压缩。40000*14*14*32

# 第二个卷积层，32连接前面的32个特征图，希望变成64个特征图
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #得到第二次池化操作后的结果40000*7*7*64，像素小但特征图多

# 全连接层densely connected layer ，把特征图全连接操作，对于分类来说变成一个向量好利用，方便机器学习。连接前面7*7*64个样本，连接后面的就是想要得到1024个维度数据，而不是全用样本数的三千多的
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024]) #1024个向量对应的b
# (40000, 7, 7, 64) => (40000, 3136) reshape方便后面使用
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #-1表示可以计算出的，用总数除以第二个7*7*64的值
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #连接到一起，得到输出40000*1024

# 将dropout加到全连接层上
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer for deep net 再来一个全连接层操作，将提取出来的1024维的特征通过w和b对应到0-9这10个类别上
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #分类操作，得到每个类别的概率。y是 (40000, 10)

# cost function 通常用交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# optimisation function 优化方式，常用梯度下降，传入学习率，用来最小化cost函数。后面会调用的优化函数
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation评估操作，看真实值和预测值的精度。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# prediction function 预测值，看那个类别的概率值最大。
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y,1)



#3运行模型，迭代运算求解——————————————
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
# serve data by batches 一个一个batch的取数据
def next_batch(batch_size):    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed   
    start = index_in_epoch
    index_in_epoch += batch_size  
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# start TensorFlow session 初始化变量，前面定义完后并没有初始化。这三行代码必用
init = tf.initialize_all_variables() #初始化所有变量
sess = tf.InteractiveSession() #指定交互的图结构：找一个可以计算的图的区域，tensorflow所有结构都是图结构
sess.run(init) #找出图结构后并没有初始化，要run

# visualisation variables 这几个list是用来做显示的，看结果
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS): 
    #get new batch 每次迭代要先拿出batch，包括数据和label
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,  #feed_dict,用来传入placeholder空缺的值
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y_: validation_labels[0:BATCH_SIZE], 
                                                            keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT}) #上面都是可视化的展示，这才是实际运行模型

# check final accuracy on validation set  画迭代过程中的精度图
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 
                                                   y_: validation_labels, 
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
    
    
