# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:20:40 2019

@author: lei
"""
import os

import librosa
import numpy as np


def MaxMinNormalization(x):
    """
    线性归一化，将输入list归一化
    :param x: list类型
    :return: 归一化list
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def normalization(list):
    """
    归一化接口，目前只支持线性归一化
    :param list: 矩阵形式
    :return: 归一化矩阵
    """
    out = []
    for x in list:
        out.append(MaxMinNormalization(x))
    return out


def get_max(list):
    """
    提取音频序列中的极大值特征
    :param list，宽度固定为20维，长度不限
    :return:20维数组
    """
    average = []
    arr_temp = np.array(list)
    # arr_temp=np.dot(arr_temp,arr_temp.T)
    for a in arr_temp:
        average.append(max(a))
    # average.append(math.atan(max(a)) * 2 / 3.1415926)
    return average


def load(file):
    """
    输入文件名，加载数据
    :param file:文件名
    :return:浮点型数组
    """
    list = []
    f = open(file, 'r', encoding='UTF-8')
    for line in f:
        line_list = line.replace(',\n', '').split(',')
        for i in range(len(line_list)):
            line_list[i] = float(line_list[i])
        list.append(line_list)
    return list


def get_data():
    """
    获取所有数据，包括音频mfcc特征数据和标签数据，一共三个人的音频数据
    x_data:[[20],[20]]
    :return: x_data,y_data
    """
    x_data = []
    y_data = []
    src_path = '刘宝瑞/'
    filename = os.listdir(src_path)
    for item in filename:  # 进入到文件夹内，对每个文件进行循环遍历
        y, sr = librosa.load(src_path + item)
        a = librosa.feature.mfcc(y=y, sr=sr)
        x_data.append(get_max(a))
        y_data.append(1)
    src_path = '姜宝林/'
    filename = os.listdir(src_path)
    for item in filename:  # 进入到文件夹内，对每个文件进行循环遍历
        y, sr = librosa.load(src_path + item)
        a = librosa.feature.mfcc(y=y, sr=sr)
        x_data.append(get_max(a))
        y_data.append(2)
    src_path = '郭德纲/'
    filename = os.listdir(src_path)
    for item in filename:  # 进入到文件夹内，对每个文件进行循环遍历
        y, sr = librosa.load(src_path + item)
        a = librosa.feature.mfcc(y=y, sr=sr)
        x_data.append(get_max(a))
        y_data.append(3)
    print("OK")
    return x_data, y_data


def shuffer(x, y):
    """
    打乱数据
    :param x: [[20],[20]]
    :param y: [[0,1,0],[1,0,0]]]onehot数据
    :return: x_out，y_out打乱的数据
    """
    x_out = []
    y_out = []
    all = []
    for i in range(0, len(y)):
        all.append([x[i], y[i]])
    import random
    random.seed(0)
    random.shuffle(all)
    for item in all:
        x_out.append(item[0])
        y_out.append(item[1])
    return x_out, y_out


def data_split(x, y, rate):
    """
    通过设定训练集和验证集的比率，来调节数据
    :param x: 输入矩阵
    :param y: 输出矩阵
    :param rate: 浮点型0-1之间
    :return: train_data，test_data
    """
    num = int(rate * len(y))
    train_data = [x[:num], y[:num]]
    test_data = [x[num:], y[num:]]
    return train_data, test_data
# x,y=shuffer([1,2,3,4,5],[1,2,3,4,5])
# print(x,y)
# x,y=get_data()
# f=open('x.txt','w',encoding='UTF-8')
# for line in x:
#     for a in line :
#         f.write(str(a)+',')
#     f.write('\n')
# f.close()
# f1=open('y.txt','r',encoding='UTF-8')
# f=open('y1.txt','w',encoding='UTF-8')
# for line in f1:
#     a =int(line)-1
#     for i in range(3):
#         if i==a:
#             f.write('1,')
#         else:
#             f.write('0,')
#     f.write('\n')
# f.close()
# f1.close()
# print("OK")
