# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas 
#1描述数据
titanic = pandas.read_csv("/Users/dustin/Documents/study/Github/Python/wheel/ml/泰坦尼克/train.csv")
titanic.head(5)
print(titanic.describe())
print(titanic["Sex"].unique())

#2预处理
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #年龄缺失值用中位数代替
# Replace all the occurences of male with the number 0. 字符型的类别变成数值型
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#3线性回归模型与验证
#线性回归与K层交叉检验
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
# The columns we'll use to predict the target 特征选取
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize our algorithm class 模型实例化
alg = LinearRegression()
# cv返回训练、测试集相对应的行索引 row indices corresponding to train and test.
# 用 random_state 保证重复试验时，每次分割能得到相同的数据集
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = [] #定义预测结果list
for train, test in kf: #train、test都是有对应的3个代表索引值的list；数据集平分成3分，每次两份训练一份验证
    # 用来训练算法的特征，只用训练集的
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train] #目标，待预测值
    alg.fit(train_predictors, train_target) #开始训练
    # 使用模型在测试集上预测
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions) #将每次预测值list放到一个大list中，方便整合，统一与真实值比对
import numpy as np
# 将3次分开的预测值list，整合成一个list 
predictions = np.concatenate(predictions, axis=0)
# 因为用线性回归，而结果是分类，所以手动调整，得出精度
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)#打印精度

#4逻辑回归
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
alg = LogisticRegression(random_state=1)
# 把多重验证的每次的精度放到arrary中 (比上面的方法简单的多)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean()) #计算均值

#5随机森林
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# 随机森林：n_estimators是决策树个数，min_samples_split 是节点允许分裂的最小样本数，min_samples_leaf 是最小叶子节点样本数
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean()) #多重验证精度平均值
# 调整随机森林参数，再次预测
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean()) #从78提升到81

#6特征工程：添加特征
#加入家庭大小和名字长度，这里的家庭大小不也是线性的？？？
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
import re
#函数：获取名字中的称谓
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))
# 将title映射成数值型，有些title太少了, 压缩到相同或类似的类里
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
# 将title加入dataframe中
titanic["Title"] = titles

#7特征工程：选取
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]
# Perform feature selection
selector = SelectKBest(f_classif, k=5) #选择排名排在前n个的特征，在高维特征的情况下，这个过程可能会非常非常慢
selector.fit(titanic[predictors], titanic["Survived"])
scores=selector.scores_# scores = -np.log10(selector.pvalues_)获取每个特征的 raw p-values 并转换成scores
# 画出分数
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
# 选取4个较好的特征
predictors = ["Pclass", "Sex", "Fare", "Title"]
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

#8两个模型综合：迭代决策树+逻辑回归
from sklearn.ensemble import GradientBoostingClassifier #ensemble库支持众多集成学习算法和模型。
import numpy as np
# 算法、参数集合
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
# 多重验证
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
# 使用两个模型，综合预测结果
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = [] #放预测值，最后和真实值比较
for train, test in kf:
    full_test_predictions = []
    # 用各个训练集训练各个模型，然后用多个模型预测测试集，放到list中
    for alg, predictors in algorithms: #遍历 算法、参数集
        alg.fit(titanic[predictors].iloc[train,:], titanic["Survived"].iloc[train])  #用不同模型训练相应实例，
        # 用模型预测测试集上的概率，需要转变类型.astype（float),防止报错
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # 用一个简单方法综合两个模型的预测结果，求平均值，
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # 连续值变离散值，闸值设在0.5，大于就是1，小于就是0
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
# 将所有预测值融合到一个list
predictions = np.concatenate(predictions, axis=0)
# 计算精度
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)


'''
以下为级联模型，即使用超过两层的模型来预测结果
'''


#导入库
#基本
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
#画图
import plotly.offline as py
import plotly.graph_objs as go
# 用一下五种模型做级联模型
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;

#1导入数据
train = pd.read_csv('/Users/dustin/Documents/study/Database/test/ml/泰坦尼克/train.csv')
test = pd.read_csv('/Users/dustin/Documents/study/Database/test/ml/泰坦尼克/test.csv')
# Store our passenger ID for easy access
PassengerId = test['PassengerId']
train.head(3)
full_data = [train, test]#全部数据集

#2预处理
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# 按是否有cabin特征分类
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# 构造FamilySize特征
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# 构造IsAlone特征
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
#  Embarked特征缺失值用众数代替
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# fare特征缺失值用中位数代替，并按数值等分成4等分
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# age按分布规律，填充缺失值，并均分为5份，离散化
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# 从姓名中提取头衔的函数
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# 构造title特征
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# 简化title特征
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#将 sex title embarked fare age 等特征变为数值型
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) 
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                              = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    
# 3特征选择
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)
train.head()
#特征关系可视化
colormap = plt.cm.viridis #colormap.翠绿色
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sb.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#4构建模型
# 稍后会使用的参数
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # 为了再现性
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
# 构造一个扩展延伸 分类器的类，方便后面对综合模型的构建
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed #初始化模型的seed
        self.clf = clf(**params) #可传入模型参数，默认为none
    #训练模型
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    #预测，返回预测值
    def predict(self, x):
        return self.clf.predict(x)
    #返回训练模型
    def fit(self,x,y):
        return self.clf.fit(x,y)
    #返回各特征重要性
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
# 获取输入的模型的预测结果，含训练集和预测集
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,)) #用0填充一个ndarray，需传入array的shape
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest)) #生成未初始化的随机值   
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        #用导入的train数据训练导入的分类器
        clf.train(x_tr, y_tr)
        #多重验证train，返回预测值，还有在测试集上的预测值list
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0) #将多重验证在测试集上的结果，去均值得到模型最终测试集的预测
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) #分别返回在训练集和测试集上的预测,reshape中表明几行几列，-1代表任意值，
# 为上述分类器输入参数
# 随机森林
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 6,
    'min_samples_leaf': 2
}
# Extra Trees 
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}
# AdaBoost 一种迭代算法
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}
# Gradient Boosting 梯度下降决策树GBDT
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}
# Support Vector Classifier 支持向量机
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
# 创建5个模型实例，定义seed，并输入参数
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# 创建各数据集 Numpy arrays 
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values 
x_test = test.values 
# 创建各模型的训练、测试集上的预测结果，通过 get_oof 函数；并准备作为新的特征
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
print("Training is complete")
#查看各模型下特征的重要性，没有支持向量机？？
rf_features = rf.feature_importances(x_train,y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train,y_train)
#获取特征的名称 ndarray
cols = train.columns.values
# 将各模型特征与特征重要值放到一个dataframe当中,这个是pandas库的
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    }) 

#5画图，常使用plotly包，本是开源的JS图表库，开放api接口
#散点图1-rf
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]
#以上为数据，以下为图层信息
layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig,filename='scatter2010a')
#散点图2-et
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]
#以上为数据，以下为图层信息
layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig,filename='scatter2010b')
# 散点图3-adaboost
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]
#以上为数据，以下为图层信息
layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig,filename='scatter2010c')
# 散点图4-gb
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]
#以上为数据，以下为图层信息
layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig,filename='scatter2010d')

#6综合模型
#看各模型预测结果的相关性
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()#预览决策结果矩阵
#查看特征重要程度的相关度
data = [ #热力图要三个坐标，此处x,y相同，表示两两关系，z表示相关度
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Portland',
            showscale=True,
            reversescale = True
    )
]
py.plot(data, filename='labelled-heatmap')#画出不同模型的特征重要程度的相关性
#第二层级的模型，x_train对应之前的y_train,可以训练各模型结果和真实值的第二层模型。然后用两层模型预测，输入x_test，预测未知的y_test
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
#第二层模型尝试构建：
from sklearn import cross_validation
rf = RandomForestClassifier(random_state=1, n_estimators=100)
kf = cross_validation.KFold(x_train.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(rf, x_train, y_train, cv=kf)
print(scores.mean()) #多重验证精度平均值
#最终预测
rf.fit(x_train, y_train)
predictions = rf.predict(x_test) 
#finished
