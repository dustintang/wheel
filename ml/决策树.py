#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:54:08 2017

@author: dustin
"""

#1导入数据______
import pandas as pd
iris_data = pd.read_csv('/Users/dustin/Documents/study/Github/Python/数据挖掘/决策树/iris.data')
iris_data.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'class']
iris_data.head()

#2观察数据__________
iris_data.describe()#数据基础描述，不包括class这种非数据类型
import matplotlib.pyplot as plt
import seaborn as sb#以matplotlib为基础，封装
sb.pairplot(iris_data.dropna(), hue='class')###多变量对比图，不能有缺失值。显示不同变量间的关系
plt.figure(figsize=(10, 10))#figure作用新建绘画窗口,独立显示绘画的图片
for column_index, column in enumerate(iris_data.columns):###小提琴图，enumerate是枚举，遍历并加上从0开始的索引
    if column == 'class':
        continue
    plt.subplot(2, 2, column_index + 1)#画多个图，前两个是行列数，最后一个参数决定画图区域
    sb.violinplot(x='class', y=column, data=iris_data)#作图，绘制小题情图

#3数据集切割________
from sklearn.cross_validation import train_test_split#多重验证中的训练集测试集划分
all_inputs = iris_data[['sepal_length_cm', 'sepal_width_cm',
                             'petal_length_cm', 'petal_width_cm']].values#dataframe[仅能传递一个参数]所以传入list。
all_classes = iris_data['class'].values#加values变成数组ndarray，不加是series还是pandas的一种数据类型
(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75, random_state=1)#多个变量同时赋值，切割操作。每个树的样本为75%，随机状态为1，可以多次对比

#4构建决策树__________
from sklearn.tree import DecisionTreeClassifier#决策树包的分类器
#  1.criterion  gini  or  entropy
#  2.splitter  best or random 前者是在所有特征中找最好的切分点 后者是在部分特征中（数据量大的时候）
#  3.max_features  None（所有），log2，sqrt，N  特征小于50的时候一般使用所有的
#  4.max_depth  数据少或者特征少的时候可以不管这个值，如果模型样本量多，特征也多的情况下，可以尝试限制下
#  5.min_samples_split  如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分
#                       如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
#  6.min_samples_leaf  这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被
#                      剪枝，如果样本量不大，不需要管这个值，大些如10W可是尝试下5
#  7.min_weight_fraction_leaf 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起
#                          被剪枝默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，
#                          或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
#  8.max_leaf_nodes 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
#                   如果加了限制，算法会建立在最大叶子节点数内最优的决策树。
#                   如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制
#                   具体的值可以通过交叉验证得到。
#  9.class_weight 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多
#                 导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重
#                 如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。
#  10.min_impurity_split 这个值限制了决策树的增长，如果某节点的不纯度
#                       (基尼系数，信息增益，均方差，绝对差)小于这个阈值
#                       则该节点不再生成子节点。即为叶子节点 。
decision_tree_classifier = DecisionTreeClassifier()#实例化，用类创建一个实例对象
# Train the classifier on the training set 训练集上训练
decision_tree_classifier.fit(training_inputs, training_classes)#训练决策树，默认返回一个完整的包含默认参数的输入函数
# Validate the classifier on the testing set using classification accuracy 测试集上验证
decision_tree_classifier.score(testing_inputs, testing_classes)#评价标准为默认的gini系数

#5多重验证_____________
from sklearn.cross_validation import cross_val_score
import numpy as np
decision_tree_classifier = DecisionTreeClassifier()#再实例化一个变量
# cross_val_score returns a list of the scores, which we can visualize 多重验证得分函数返回一个可视化得分list
# to get a reasonable estimate of our classifier's performance 优化分类预测的结果
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10)#还有scoring参数可以选择评价标准，cv为多重验证划分数据集个数
sb.distplot(cv_scores)#集合了matplotlib的hist()与核函数估计kdeplot的功能，增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途, 可加kde=False, rug=True
plt.title('Average score: {}'.format(np.mean(cv_scores)))#因为seaborn是封装的，所以底层的plt函数也可以叠加

#6剪枝____________
decision_tree_classifier = DecisionTreeClassifier(max_depth=1)#预剪纸，最大深度为1
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10)#预剪枝因为深度为1，gini系数较低
sb.distplot(cv_scores, kde=False)
plt.title('Average score: {}'.format(np.mean(cv_scores)))
#不同剪枝策略的评分
from sklearn.grid_search import GridSearchCV #专门调试参数的函数。实现了fit，predict，predict_proba等方法，并通过交叉验证对参数空间进行求解，寻找最佳的参数。 
from sklearn.cross_validation import StratifiedKFold #cv参数就是代表不同的cross validation的方法了。如果cv是一个int数字的话，并且如果提供了raw target参数，那么就代表使用StratifiedKFold分类方式，如果没有提供raw target参数，那么就代表使用KFold分类方式。
decision_tree_classifier = DecisionTreeClassifier() #重新实例化
parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]} #调试参数的剪枝策略
cross_validation = StratifiedKFold(all_classes, n_folds=10) #是一种将数据集中每一类样本的数据成分，按均等方式拆分的方法。
grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation) #调试参数，剪枝策略和多重验证cv方法传入
grid_search.fit(all_inputs, all_classes) #构建决策树
print('Best score: {}'.format(grid_search.best_score_)) #字符串格式化，返回最佳gimi值
print('Best parameters: {}'.format(grid_search.best_params_)) #返回最佳gimi值的剪枝策略参数
#不同剪枝策略的评分可视化
grid_visualization = []
for grid_pair in grid_search.grid_scores_:
    grid_visualization.append(grid_pair.mean_validation_score) #将不同剪枝策略的多重验证平均得分放入一个list
grid_visualization = np.array(grid_visualization) #变为数组ndarray
grid_visualization.shape = (5, 4) 
sb.heatmap(grid_visualization, cmap='Blues')#seaborn的热力图，下面为画图相关
plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'][::-1])
plt.xlabel('max_features')
plt.ylabel('max_depth')
#找到最佳预测的剪枝分类器（模型）
decision_tree_classifier = grid_search.best_estimator_
decision_tree_classifier

#7导出最佳分类器_____________
import sklearn.tree as tree
#分类标准写到文本中
with open('iris_dtc.dot', 'w') as out_file:
    out_file = tree.export_graphviz(decision_tree_classifier, out_file=out_file)
#http://www.graphviz.org/
from sklearn.externals.six import StringIO #导出pdf等文件用
import pydot
dot_data = StringIO() 
tree.export_graphviz(decision_tree_classifier, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("iris.pdf") 

#8随机森林________________
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier() #随机森林分类器实例化
parameter_grid = {'n_estimators': [5, 10, 25, 50], #在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。 
                  'criterion': ['gini', 'entropy'],
                  'max_features': [1, 2, 3, 4],  #随机森林允许单个决策树使用特征的最大数量
                  'warm_start': [True, False]}  #热启动，如果为True，则使用上次调用该类的结果然后增加新的。如果为False，则新建新的forest，默认为False
cross_validation = StratifiedKFold(all_classes, n_folds=10)#同为多重验证的方法
grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)  #多个待检验比较的模型集
grid_search.fit(all_inputs, all_classes) #训练数据集
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
grid_search.best_estimator_
