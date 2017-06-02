#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 15:49:46 2017

@author: dustin
"""

import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
#1__numpy______
vector=np.array([1,2,3,5])
matrix=np.array([[1,2,6],[2,3,1],[2,2,2],[1,2,2]])
matrix.shape#行列数
vector.dtype.name#统一的存储格式
matrix.size#数据量
matrix[0]
matrix[:,0:2]
matrix==2
a=(matrix==1) | (matrix==2)#或
b=(matrix==1) & (matrix==2)#与
matrix[(matrix==1) | (matrix==2)]#筛选
vector.astype(str)#转换存储类型
matrix.max()#最大值
matrix.sum(axis=0)#按行或列求和
c=np.arange(15).reshape(3,5)#自动生成并赋予维度
c=np.arange(2,10,1)#自动生成range
c=np.random.random((3,5))
c.ndim#维度数
matrix.ravel()#将矩阵变成向量
matrix.T#转置
np.vstack((a,b))#按行拼接
np.hstack((a,b))#按列拼接
np.hsplit(a,3)#按列等分成3份
c=a.view()#c是a的复制，但两个不想等，浅层复制，不推荐使用，推荐用copy
d=a.copy()#让d称为a的初始值，不能用等号。
matrix.argmax(axis=0)#返回制定维度的最大值的索引值
np.tile(matrix,(2,2))#将矩阵横向纵向复制
np.sort(a)#排序
np.argsort(a)#排序索引值

d=np.zeros((3,5))#初始化矩阵
e=np.ones((2,3,5),dtype=np.int32)#初始化矩阵，并定义格式
f=np.linspace(0,2*pi,100)#起点，终点，元素个数
np.sin(1)#函数可嵌套np格式：exp自然数次方，sqrt开方，floor取整，
##对维度相同的数组变换
c-d
c+1
c**2
c>0.5
a*matrix#求内积，对应位置相乘
a.dot(b)#用a乘以b，a的行乘以b的列
np.dot(a,b)#同上





#2__pandas__________
file=pd.read_csv('api/end的副本.csv')#得到的格式是dataframe，里面每一列dtype允许不同格式，但每列统一。类似矩阵
file.head(3)#头部信息，默认5行
file.tail(3)#尾部信息
file.shape#行列数
file.columns#查看列名,得到的是pandas.indexes.base.Index，不是list
file.loc[1,'lng']#索引需要加loc函数
file[['lng','lat']]#拿列数据，需要将列名放到一个list中
file.columns.tolist()#将列名变成list
#基本运算都是对每个数据单独操作的
file['lng'].max()#该列最大值
z=file.sort_values('lng',inplace=True,ascending=False)#默认是升序，不替换原表
z.reset_index(drop=True)#????重新定义索引值
a=pd.isnull(file['lng'])#判断列是否为缺失值，有缺失值难以进行直接计算
len(file[a])#缺失值个数
file['lng'].mean()#过滤掉缺失值求平均值
file.pivot_table(index='range',values='lng',aggfunc=np.mean)#统计对应于不同index的values的值的aggfunc函数，不写函数的话默认为求均值。输入的列名也可以多列的list。
file.apply()#自定义函数，可以将一长串函数浓缩到一个函数，然后apply

file.dropna(axis=0,subset=['lng'])#去掉指定列的缺失值行数据
file.dropna(axis=1)#去掉有缺失值的数据
x=pd.Series( file.lng , index=file.id)#dataframe的一行一列或几行几列都是series的结构。可以将别的变量变为索引值，当然数字索引同样有用




#3——————————————————
from sklearn.preprocessing import StandardScaler
data=pd.read_csv('/Users/dustin/Documents/study/数据团/欺诈检测/creditcard.csv')
count_classes=pd.value_counts(data['Class'],sort=True).sort_index()
count_classes.plot(kind='bar')
plt.title('fraud class')#要和上行一起执行，不能分开
#预处理
data['normAmount']=StandardScaler().fit_transform(data['Amount'].reshape(-1,1))#标准化
data=data.drop(['Time','Amount'],axis=1)#去掉列
x = data.ix[:,data.columns!='Class']#因素列集合
y = data.ix[:,data.columns=='Class']#标签列集合
#下采样
number_records_fraud=len(data[data.Class==1])#取出class等于1的行，然后计算长度
fraud_indices=np.array(data[data.Class==1].index)#取出class等于1的行的索引值，并将它们变为array
normal_indices=data[data.Class==0].index#取出class等于0的行的索引值,随机选择的样本池
random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)#在a样本中选择b个，不代替
under_sample_indices=np.concatenate([fraud_indices , random_normal_indices])#研究用数据的index值合集
under_sample_data=data.iloc[under_sample_indices,:]#新样本数据集合
x_undersample = under_sample_data.ix[:,under_sample_data.columns !='Class']
y_undersample = under_sample_data.ix[:,under_sample_data.columns =='Class']
from sklearn.cross_validation import train_test_split#交叉验证模块
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state=0)#交叉验证数据集,是随机洗牌后切割，random_state后的值代表随机洗牌的一个方法，防止每次随机不同对效果的影响
x_train_undersample, x_test_undersample,y_train_undersample, y_test_undersample= train_test_split(x_undersample,y_undersample,test_size=0.3 , random_state=0)#下采样数据集切分

##模型评估方法
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report

def printing_Kfold_scores(x_train_data,y_train_data):
    fold=KFold(len(y_train_data),5,shuffle=False)
    c_param_range=[0.01,0.1,1,10,100]  #正则惩罚项的权重
    results_table=pd.DataFrame(index=range(len(c_param_range),2),colums=['C_parameter','Mean recall '])

    

#过采样

len(file[file('range==10')])
file[file('range==10')].index
#np.random.choice(什么地方,随机选多少个，replace=False)不代替
np.concatenate([])#合并



























