from __future__ import print_function

import time
import sys
import File2Dataset
import NeuralNetwork
import numpy
from openpyxl import load_workbook
import xlsxwriter 
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from File2Dataset import import_data

savepath = 'E:\\code\\python\\6.05model\\model.ckpt'

max_steps = 100000000
batch_size = 100
learning_rate = 0.001

if __name__ == '__main__':
# def main():
    #nn = NeuralNetwork.NeuralNetwork(setting = [12,20,10,4], batchsize = batch_size, learning_rate = learning_rate)
    nn = NeuralNetwork.NeuralNetwork(setting = [12,48,48,4], batchsize = batch_size, learning_rate = learning_rate)
    nn.Create()
    nn.LoadNetwork(savepath)

    #output = nn.TestSingle(numpy.concatenate(([960*1e-5, 31281*1e-7, 0.0*1e-2, 552*1e-5], 
                                        #File2Dataset.dense_to_one_hot_single(0,4),
                                        #File2Dataset.dense_to_one_hot_single(0,2), 
                                        #File2Dataset.dense_to_one_hot_single(1,2))), 
                                        # None)

    arr1 = import_data('E:\\code\\python\\qoe_model\\8wdata_test.xlsx')
    

    j = 0
    workbook = xlsxwriter.Workbook('E:\\code\\python\\qoe_model\\result.xlsx') 
    worksheet = workbook.add_worksheet('Sheet1') 

    for i in range(599):
        a = [arr1[i,j]*1e-4, arr1[i,j + 1]*1e-5, 1, arr1[i,j + 2], arr1[i,j + 3]*1e-4, arr1[i,j + 4]*1e-3, arr1[i,j + 5]*1e-6, arr1[i,j + 6]*1e-4, arr1[i,j + 7]*1e-4, arr1[i,j + 8]*1e-7, arr1[i,j + 9]*1e-4,arr1[i,j + 10]*1e-5]
        b = [arr1[i,11], arr1[i,12], arr1[i,13], arr1[i,14]]
        output = nn.TestSingle(a)

        
        worksheet.write(i+2, j, output[0]*5)
        worksheet.write(i+2, j+1, arr1[i,11])

        worksheet.write(i+2, j+3, output[1]*5)
        worksheet.write(i+2, j+4, arr1[i,12])

        worksheet.write(i+2, j+6, output[2]*5)
        worksheet.write(i+2, j+7, arr1[i,13])

        worksheet.write(i+2, j+9, output[3]*5)
        worksheet.write(i+2, j+10, arr1[i,14])
    workbook.close()

    #print("vMOS = %f, QualityScore = %f, StallingScore = %f, LoadingScroe = %f" %(output[0]*5, output[1]*5, output[2]*5, output[3]*5))
    

    nn.Close()
  