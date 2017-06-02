#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 23:32:34 2017

@author: dustin
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cross_validation import train_test_split
#画图函数，将数据点与分类标准做于二维图上。传入数据集与分类器
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 设置画图所用参数
    markers = ('s', 'x', 'o', '^', 'v') #标记
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') #颜色
    cmap = ListedColormap(colors[:len(np.unique(y))]) #函数用来通过一个指定的颜色列表生成习惯的彩色图。将colors中自定的颜色，赋给需要颜色数量的y
    # 画出决策面decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 #上下左右空出1的距离
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))#生成网格矩阵，并定义为传入的分辨率0.02
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #传入的分类器已经训练好了，可以直接用来预测。这里预测的就是每一对坐标，因为svm就是对空间的分割
    Z = Z.reshape(xx1.shape) #再将z从一对一对的坐标换成网格点的shape
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) #函数用来画等高线并填充颜色
    plt.xlim(xx1.min(), xx1.max()) #以上的min，max都是用于转换出画图数据，这里是直接对应画图的长宽
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)): #画出每一个样本点，按类别用不同颜色和标记
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx: #暂未用上，用来高粱的画出测试集数据
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

#导入数据
iris = datasets.load_iris() # 由于Iris是很有名的数据集，scikit-learn已经原生自带了。
X = iris.data[:, [1, 2]] #为了画图方便，就用两个特征值来分类
y = iris.target # 标签已经转换成0，1，2了
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试

# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #标准化类的实例化
sc.fit(X_train) # 估算每个特征的平均值和标准差
sc.mean_ # 查看特征的平均值，由于Iris我们只用了两个特征，所以结果是array([ 3.82857143,  1.22666667])
sc.scale_ # 查看特征的标准差，这个结果是array([ 1.79595918,  0.77769705])
X_train_std = sc.transform(X_train) #通过训练集训练得到的标准化方法，来改变原始数据集，并在下面改变测试集的特征
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std)) #将训练集与测试集的特征纵向连接
y_combined = np.hstack((y_train, y_test)) # 将分类信息横向连接为一个array

# 导入SVC
from sklearn.svm import SVC
svm1 = SVC(kernel='linear', C=0.1, random_state=0) # 支持向量机实例化，用线性核，尝试不同的c值，即对于不同的松弛因子
svm1.fit(X_train_std, y_train)
svm2 = SVC(kernel='linear', C=10, random_state=0) # 用线性核
svm2.fit(X_train_std, y_train)
#画图
fig = plt.figure(figsize=(10,6)) #一整个图标就是一个figure
ax1 = fig.add_subplot(1,2,1) #子图1
plot_decision_regions(X_combined_std, y_combined, classifier=svm1) #将参数传入自定义的画图函数
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('C = 0.1')
ax2 = fig.add_subplot(1,2,2) #子图2
plot_decision_regions(X_combined_std, y_combined, classifier=svm2) #将子图2的参数传入自定义的画图函数
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('C = 10')
plt.show() #显示图像

#使用高斯核函数kernel，测试不同gamma 值
svm1 = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1.0) # 令gamma参数中的x分别等于0.1和10。代表次方数？
svm1.fit(X_train_std, y_train) 
svm2 = SVC(kernel='rbf', random_state=0, gamma=10, C=1.0) 
svm2.fit(X_train_std, y_train) 
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(1,2,1)
plot_decision_regions(X_combined_std, y_combined, classifier=svm1)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('gamma = 0.1')
ax2 = fig.add_subplot(1,2,2)
plot_decision_regions(X_combined_std, y_combined, classifier=svm2)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title('gamma = 10')
plt.show() #显示图像
