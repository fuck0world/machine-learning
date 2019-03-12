#coding=utf-8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection  import GridSearchCV
#from sklearn import cross_validation, metrics
import sys 
import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.metrics import  accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split

raw_datapath = 'E:\\code\\python\\qoe_model\\raw_data\\3w_data.csv'
data = pd.read_csv(raw_datapath)

column_names = ['InitialBufferTime', 'VideoPlayDuration','StallingRatio', 'VIDEO_BITRATE', 'VIDEO_CLARITY', 'VIDEO_ALL_PEAK_RATE', 
                'VIDEO_AVERAGE_RATE', 'USERBUFFERTIME', 'VIDEOSIZE', 'SCREEN_RESOLUTION_LONG', 'VIDEO_BUFFERING_PEAK_RATE', 
                'EVMOS', 'ELOADING', 'ESTALLING', 'USER_SCORE']
#########################################################
############ 将 name 列的离散数据进行编号 ###############
#########################################################
def class_normalization(name, X):
    
    # name不是list,是str
    a = X[name]
    b = a.value_counts()
    c = b.index

    list1 = []
    list2 = []
    for i in range(len(c)):
        list1.append(i)
        list2.append(c[i])
        
    b = a.replace(list2, list1)
    
    data1 = X.drop([name], axis=1)
    data1.insert(2, name, b)
    
    return data1

##########################################################
#################### 移除 name 列 ########################
##########################################################
def remove_col(name, all_name):
    
    list = []
    for i in range(len(column_names)):
        if column_names[i] != name:
            list.append(column_names[i])
    return list

# 生成每一个batch
def generatebatch(X,Y,n_examples, batch_size): 
    for batch_i in range(n_examples // batch_size): 
        start = batch_i * batch_size 
        end = start + batch_size 
        batch_xs = X[start:end] 
        batch_ys = Y[start:end] 
        yield batch_xs, batch_ys 

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

name = 'VIDEO_CLARITY'
data1 = class_normalization(name, data)
data1 = shuffle(data1)
data1 = data1.reset_index(drop = True)

X1 = data1[remove_col(name, column_names)]
X2 = data1[[name]]

# 选取第i个分数
Y1 = data1[column_names[11]]
Y2 = data1[column_names[12]]
Y3 = data1[column_names[13]]
Y4 = data1[column_names[14]]

scaler = MinMaxScaler()
X1_data = scaler.fit_transform(X1)
X2_data = OneHotEncoder().fit_transform(X2.values.reshape(-1, 1)).todense()
X_data = np.hstack((X1_data, X2_data)).getA()

# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
X1_train, X1_test, y1_train, y1_test = train_test_split(X_data, Y1, test_size=0.25, random_state = 33)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_data, Y2, test_size=0.25, random_state = 33)
X3_train, X3_test, y3_train, y3_test = train_test_split(X_data, Y3, test_size=0.25, random_state = 33)
X4_train, X4_test, y4_train, y4_test = train_test_split(X_data, Y4, test_size=0.25, random_state = 33)

y1_train.as_matrix()
y2_train.as_matrix()
y3_train.as_matrix()
y4_train.as_matrix()

#确定调优参数
parameters = {'n_estimators':[50,100,150], 'learning_rate':[0.5,1,1.5], 'max_depth':[1,2,3]}
#构建模型，调优,确定十折交叉验证
estimator = GradientBoostingClassifier(random_state=42)
best_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=10).fit(X1_train, y1_train)
best_clf.grid_scores_, best_clf.best_params_, best_clf.best_score_

joblib.dump(best_clf, "train_model_1.m")
GBDT = joblib.load("train_model_1.m")
y_pred = GBDT.predict(X1_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y1_test, y_pred))