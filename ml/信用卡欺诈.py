#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:56:22 2017

@author: dustin
"""

#1导入数据______________
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("/Users/dustin/Documents/study/Github/Python/数据挖掘/欺诈检测/creditcard.csv")
data.head()

#2观察数据_____________
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

#3数据预处理____________
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()

#4样本采样_____________
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']
# Number of data points in the minority class 少数标签的个数（不均衡样本）
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index) #少数标签在数据集中的索引值
# Picking the indices of the normal classes 多数标签在数据集中的索引值
normal_indices = data[data.Class == 0].index #没变成ndarray，而是pandas.indexes格式
# Out of the indices we picked, randomly select "x" number (number_records_fraud) 下采样，随机采样
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False) #下采样，从样本中不放回的抽出和少数标签相同数量的多数标签
random_normal_indices = np.array(random_normal_indices)
# Appending the 2 indices 下采样后，样本的所有索引
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices]) #下采样索引array
# Under sample dataset 下采样数据集
under_sample_data = data.iloc[under_sample_indices,:] #所用到的数据集
#下采样的切割数据集
X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

#5训练测试数据集切割
from sklearn.cross_validation import train_test_split
# Whole dataset 所有切割的数据集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# Undersampled dataset 下采样切割数据集
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)

#6参数选取函数（惩罚项）__________________
#评价指标Recall = TP/(TP+FN)
from sklearn.linear_model import LogisticRegression #线性模型中的逻辑回归
from sklearn.cross_validation import KFold, cross_val_score  #交叉验证，k-折交叉验证：将数据集A随机分为k个包，每次将其中一个包作为测试集，剩下k-1个包作为训练集进行训练。／ cross_val_score测试交叉验证模型可靠性，回归与分类都可以用
from sklearn.metrics import confusion_matrix,recall_score,classification_report #度量标准包,classification_report为文字，包含精准度，recall，权重，以及对精准度和recall的一种加权平均的f1分数。展现模型的好坏
#输入训练集，导出多个惩罚项的模型的交叉验证分数，并选出最好的惩罚项参数。都是在测试集上做的
def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 
    # Different C parameters 不同的惩罚力度
    c_param_range = [0.01,0.1,1,10,100]
    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range
    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')
        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')    
    return best_c
#调用函数，测试模型
best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)

#7画混淆矩阵函数____________
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): #函数画出混淆矩阵，传入混淆矩阵和标签种类这里是0和1
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    #画图核心数据
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #画图相关
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#8绘制下采样训练出的模型在下采样测试集上的混淆矩阵_________
import itertools
lr = LogisticRegression(C = best_c, penalty = 'l1') #逻辑回归模型采用最佳的惩罚项权重best_c
lr.fit(X_train_undersample,y_train_undersample.values.ravel()) #用下采样数据集训练
y_pred_undersample = lr.predict(X_test_undersample.values) #用训练好的模型测试测试集上的数据
# Compute confusion matrix 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample) #比对预测值和真实值的混淆矩阵
np.set_printoptions(precision=2) #设置打印选项，位数精准度
#计算测试集recall
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# Plot non-normalized confusion matrix #画出非标准化混淆矩阵
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix') #传递混淆矩阵参数到画图函数中
plt.show()

#9绘制下采样训练出的模型在全样本测试集上的混淆矩阵_________
lr = LogisticRegression(C = best_c, penalty = 'l1') #使用最佳惩罚权重，惩罚用L1正则化：+abs（w）
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values) #用模型测试整个样本的的测试集
# Compute confusion matrix 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

#10全样本下的最佳惩罚项权重_________
best_c = printing_Kfold_scores(X_train,y_train)

##11绘制全样本训练出的模型在全样本测试集上的混淆矩阵_________
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,y_train.values.ravel())
y_pred_undersample = lr.predict(X_test.values)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred_undersample)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

#12调整逻辑回归sigmoid函数的预值thresholds_______________
lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values) #下采样训练
#0-1之间的预值测试
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#新建画图
plt.figure(figsize=(10,10))
j = 1
for i in thresholds: #画出不同预值下的混淆矩阵
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i
    #画图位置
    plt.subplot(3,3,j)
    j += 1
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 

#13使用过采样获得更加可靠的模型_____________________
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#读取数据
credit_cards=pd.read_csv('/Users/dustin/Documents/study/Github/Python/数据挖掘/欺诈检测/creditcard.csv')
columns=credit_cards.columns
# The labels are in the last column ('Class'). Simply remove it to obtain features columns 分开标签列和特征列
features_columns=columns.delete(len(columns)-1)
features=credit_cards[features_columns]
labels=credit_cards['Class']
#切割数据
features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.2, 
                                                                            random_state=0)
#过采样，采用smote样本生成策略
oversampler=SMOTE(random_state=0)#随机生成样本实例化
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)#smote函数，导入训练用的特征和标签数据（应该是随机生成索引值，并不影响样本本身，只是重复的方法）
len(os_labels[os_labels==1])#过采样的样本数量
#选取过采样的最佳惩罚项
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
best_c = printing_Kfold_scores(os_features,os_labels)
#使用最佳惩罚项，并对全样本预测
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(os_features,os_labels.values.ravel())
y_pred = lr.predict(features_test.values)
# Compute confusion matrix 计算混淆矩阵
cnf_matrix = confusion_matrix(labels_test,y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show() #画出过采样训练集得到的模型在全样本下的混淆矩阵
