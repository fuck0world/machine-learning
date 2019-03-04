import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd

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

from sklearn.utils import shuffle

name = 'VIDEO_CLARITY'
data1 = class_normalization(name, data)
data1 = shuffle(data1)
data1 = data1.reset_index(drop = True)

#X = data1[column_names[0:11]]
X1 = data1[remove_col(name, column_names)]
X2 = data1[[name]]

Y = data1[column_names[11:15]]
Y1 = data1[column_names[11]]

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()

X1_data = scaler.fit_transform(X1)
X2_data = OneHotEncoder().fit_transform(X2.values.reshape(-1, 1)).todense()
X_data = np.hstack((X1_data, X2_data))

[raw, col] = X_data.shape
shape = 8*8
X_zero = np.zeros([raw, shape - col])

X_data = np.hstack((X1_data, X2_data, X_zero)).getA()

#one-hot编码
Y_data = OneHotEncoder().fit_transform(Y).todense().getA()
Y1_data = OneHotEncoder().fit_transform(Y1.values.reshape(-1, 1)).todense().getA()

from sklearn.model_selection import train_test_split
# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=33)

X = X_data.reshape(-1,8,8,1)

# 使用MBGD算法，设定batch_size为8
batch_size = 640
 
# 生成每一个batch
def generatebatch(X,Y,n_examples, batch_size): 
    for batch_i in range(n_examples // batch_size): 
        start = batch_i * batch_size 
        end = start + batch_size 
        batch_xs = X[start:end] 
        batch_ys = Y[start:end] 
        yield batch_xs, batch_ys 

tf.reset_default_graph()

# 输入层
tf_X = tf.placeholder(tf.float32,[None,8,8,1])
tf_Y = tf.placeholder(tf.float32,[None,5])

# 卷积层+激活层 
conv_filter_w1 = tf.Variable(tf.random_normal([3, 3, 1, 5])) 
conv_filter_b1 = tf.Variable(tf.random_normal([5])) 
relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(tf_X, conv_filter_w1,strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1)

# 池化层
max_pool1 = tf.nn.max_pool(relu_feature_maps1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 卷积层 
conv_filter_w2 = tf.Variable(tf.random_normal([3, 3, 5, 5])) 
conv_filter_b2 = tf.Variable(tf.random_normal([5])) 
conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 2, 1], padding='SAME') + conv_filter_b2 

# BN归一化层+激活层 
batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True) 
shift = tf.Variable(tf.zeros([5])) 
scale = tf.Variable(tf.ones([5])) 
epsilon = 1e-3 
BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon) 
relu_BN_maps2 = tf.nn.relu(BN_out)

# 池化层
max_pool2 = tf.nn.max_pool(relu_BN_maps2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 将特征图进行展开
max_pool2_flat = tf.reshape(max_pool2, [-1, 2*2*5])

# 全连接层 
fc_w1 = tf.Variable(tf.random_normal([2*2*5,50])) 
fc_b1 = tf.Variable(tf.random_normal([50])) 
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)

# 输出层 
out_w1 = tf.Variable(tf.random_normal([50,5])) 
out_b1 = tf.Variable(tf.random_normal([5])) 
pred = tf.nn.softmax(tf.matmul(fc_out1, out_w1)+out_b1)

loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(pred,1e-11,1.0)))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_pred = tf.argmax(pred,1)
bool_pred = tf.equal(tf.argmax(tf_Y,1),y_pred)
# 准确率
accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32))

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    for epoch in range(1000): 
        # 迭代1000个周期 
        for batch_xs,batch_ys in generatebatch(X,Y1_data,Y1_data.shape[0],batch_size): 
            # 每个周期进行MBGD算法 
            sess.run(train_step, feed_dict={tf_X:batch_xs,tf_Y:batch_ys}) 
        if(epoch%100==0): 
            res = sess.run(accuracy,feed_dict={tf_X:X,tf_Y:Y1_data}) 
            print((epoch,res))
    res_ypred = y_pred.eval(feed_dict={tf_X:X,tf_Y:Y1_data}).flatten() 
    # 只能预测一批样本，不能预测一个样本 
    print(res_ypred)

from sklearn.metrics import  accuracy_score
#print(accuracy_score(y_train,res_ypred.reshape(-1,1)))