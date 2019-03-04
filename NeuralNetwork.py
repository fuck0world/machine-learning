# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import numpy as np

class NeuralNetwork(object):

    BN_DECAY = 0.999
    BN_EPSILON = 1e-3
    #0.00004
    WEIGHT_DECAY = 1e-4 
    NET_VARIABLES = 'network_variables'
    
    #def __init__(self, setting = [6,10,5,3], batchsize = 128, activatefun=tf.nn.sigmoid, learning_rate = 0.01 ):
    def __init__(self, setting, batchsize = 128, activatefun=tf.nn.sigmoid, learning_rate = 0.01 ):

        self.network_setting = setting
        self.network_activatefun = activatefun
        self.network_train_batchsize = batchsize
        self.network_learning_rate = learning_rate
        
        # placeholder()函数是在神经网络构建graph的时候在模型中的占位
        self.input_placeholder = tf.placeholder(tf.float32, shape=( self.network_train_batchsize, self.network_setting[0] ))
        self.output_placeholder = tf.placeholder(tf.float32, shape=( self.network_train_batchsize, self.network_setting[-1] ))
 
        #self.network_output = self._Create(self.input_placeholder, self.output_placeholder, activatefun)
        
    def _get_variable(self, name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
        
        #规则化
        if weight_decay > 0:
        # TensorFlow会将L2的正则化损失值乘以weigh_dacay使得求导得到的结果更加简洁
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        # tf.GraphKeys.GLOBAL_VARIABLES是默认的 Variable 对象集合，
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.NET_VARIABLES]
        return tf.get_variable(name,shape=shape,initializer=initializer,dtype=dtype,regularizer=regularizer,collections=collections,trainable=trainable)

    def Create(self):
        
        input = self.input_placeholder
        output = self.output_placeholder
        activation = self.network_activatefun
        
        x = input
        for i in range(len(self.network_setting)-1):
            #tf.variable_scope变量共享域
            with tf.variable_scope('Layer%d' % (i + 1)):
                initializer = tf.truncated_normal_initializer( stddev=math.sqrt(1.0/float(self.network_setting[i])) )
                weights = self._get_variable('weights',
                                        shape=[self.network_setting[i], self.network_setting[i+1]],
                                        dtype='float',
                                        initializer=initializer,
                                        weight_decay=self.WEIGHT_DECAY)
        
                biases = self._get_variable('biases', weights.get_shape()[-1:],initializer=tf.zeros_initializer)
                x = activation(tf.matmul(x, weights) + biases)
                
        self.network_output = x
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(x-output),reduction_indices=[1]) , name='mse')
        #self.loss = tf.reduce_mean(tf.reduce_sum(tf.log(tf.abs(x-output)+1),reduction_indices=[1]) , name='mse')
        
        #optimizer = tf.train.GradientDescentOptimizer(self.network_learning_rate)
        optimizer = tf.train.AdagradOptimizer(self.network_learning_rate)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        
        #保存模型
        self.saver = tf.train.Saver()
        self._initialize()
        
        return x
    
    def _initialize(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    
    def SaveNetwork(self,train_dir):
        self.saver.save(self.sess, train_dir)
        
        
    def LoadNetwork(self,modelfilepath):
        self.saver.restore(self.sess, modelfilepath)
        #self.saver.save(self.sess, train_dir)
    
    def TestBatch(self, inputbatch, outputbatch = None):
        if(outputbatch is None):
            return self.sess.run([self.network_output],
                               feed_dict={self.input_placeholder:inputbatch})
        else:
            return self.sess.run([self.network_output, self.loss],
                               feed_dict={self.input_placeholder:inputbatch, self.output_placeholder:outputbatch})
        
    def TestSingle(self, input, output = None):
        inputbatch = np.zeros([self.network_train_batchsize,self.network_setting[0]])
        inputbatch[0,:] = input
        if(output is None):
            outputbatch_e = self.sess.run(self.network_output,
                               feed_dict={self.input_placeholder:inputbatch})
            return outputbatch_e[0,:]
        else:
            outputbatch = np.zeros([self.network_train_batchsize,self.network_setting[0]])
            outputbatch[0,:] = output
            outputbatch_e, loss_value = self.sess.run([self.network_output, self.loss],
                               feed_dict={self.input_placeholder:inputbatch, self.output_placeholder:outputbatch})
            return outputbatch_e[0,:], loss_value*self.network_train_batchsize
            
        
    def TrainBatch(self, inputbatch, outputbatch):
        _, loss_value = self.sess.run([self.train_op, self.loss],
                               feed_dict={self.input_placeholder:inputbatch, self.output_placeholder:outputbatch})
        return loss_value
    
    def Close(self):
        self.sess.close()
        