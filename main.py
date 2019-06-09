# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:20:40 2019

@author: lei
"""
import BP
import datahelper as dh

if __name__ == '__main__':
    bp = BP.BPNeuralNetwork()
    x = dh.load('datas/x.txt')
    y = dh.load('datas/y.txt')
    x, y = dh.shuffer(x, y)
    x = dh.normalization(x)
    train_list, test_list = dh.data_split(x, y, 0.5)
    bp.train(train_list, test_list, 100, 0.05, 0.1)
    bp.save('%f.h7' % (bp.test_acc))
