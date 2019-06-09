# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:20:40 2019

@author: lei
"""

import math
import pickle
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0  # 输入层数量
        self.hidden_n = 0  # 隐藏层
        self.output_n = 0  # 输出层
        self.input_cells = []  # 输入矩阵
        self.hidden_cells = []  # 隐藏层矩阵
        self.output_cells = []  # 输出矩阵
        self.input_weights = []  # 输入权重矩阵
        self.output_weights = []  # 输出权重矩阵
        self.input_correction = []  # 输入矫正矩阵
        self.output_correction = []  # 输出矫正矩阵
        self.train_acc = .0  # 训练集准确率
        self.test_acc = .0  # 测试集准确率

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n  # 输入层节点初始化
        self.hidden_cells = [1.0] * self.hidden_n  # 隐藏层节点初始化
        self.output_cells = [1.0] * self.output_n  # 输出层节点初始化
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        # 输入层权重初始化
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # 输出层权重初始化

        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def get_precise_rate(self, cases, labels):
        right_num = 0
        length = len(labels)
        for i in range(length):
            predict = self.predict(cases[i])
            predict.index(max(predict))
            if predict.index(max(predict)) == labels[i].index(max(labels[i])):
                right_num += 1
        return right_num / length

    def train(self, train_list, test_list, limit=10000, learn=0.05, correct=0.1):
        self.train_acc = .0
        self.test_acc = .0
        for j in range(limit):
            error = 0.0
            for i in range(len(train_list[0])):
                label = train_list[1][i]
                case = train_list[0][i]
                error += self.back_propagate(case, label, learn, correct)
                # print("第%d轮，训练次数：%d"%(j,i),end='\r')
            self.train_acc = self.get_precise_rate(train_list[0], train_list[1])
            self.test_acc = self.get_precise_rate(test_list[0], test_list[1])
            print("训练次数：%d,train_acc:%f,test_acc%f" % (j, self.train_acc, self.test_acc), end='\r')

    def save(self, fliename):
        f = open(fliename, 'wb')
        pickle.dump(self, f)
        f.close()

    def load(self, fliename):
        f = open(fliename, 'rb')

        return pickle.load(f)

    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))

# if __name__ == '__main__':
#     nn = BPNeuralNetwork()
#     nn.setup(20,200,3)
#     nn.train(x)
#     nn.test()
