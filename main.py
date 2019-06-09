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
    f = open('qushi.txt', 'w', encoding='UTF-8')
    for hide in range(3, 50):

        bp.setup(20, hide, 3)
        i = 0
        print("第%d轮" % (hide - 2))
        for i in range(2000):
            bp.train(train_list, test_list, 1, 0.05, 0.1)
            i += 1
            if bp.test_acc > 0.99:
                f.write(str(hide) + '\t' + str(i) + '\n')
                break
        if i >= 2000:
            f.write(str(hide) + '\t' + '---' + '\n')
        # bp.save('%f.h7'%(bp.test_acc))
    f.close()
