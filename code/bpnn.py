from math import *
import random
import matplotlib.pyplot as plt
import numpy as np
import time

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmod_derivate(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
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
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells[h]) * error
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

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for i in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        cases = []
        labels = []
        self.setup(85, 11, 1)
        #use dataset
        train_set = open('../data/train.txt', 'r')
        #train_set = open('spiral.txt', 'r')
        lines = train_set.readlines()
        for line in lines:
            sample = line.replace('\t\n', '').split('\t')
            vector = []
            label = []
            for j in range(len(sample)-1):
                vector.append(float(sample[j]))
            label.append(float(sample[-1]))
            cases.append(vector)
            labels.append(label)
        
        #compute dataset
        # d=1/7;c=32;n=2
        # for i in range(int(c*n)+1):
	    #     t=i*2*pi/c+0.5*pi
	    #     a=d/(2*pi)
	    #     r=a*t
	    #     x=r*cos(t)
	    #     y=r*sin(t)
	    #     cases+=[[x+0.5,y+0.5],[-x+0.5,-y+0.5]]
	    #     labels+=[[0],[1]]  

        print("Start train...")
        start = time.clock()
        self.train(cases, labels, 200000, 0.08, 0.1)
        end = time.clock()
        print("Training complete! Time used: %.2f s"%(end - start))

        # for i in np.arange(-6,6,0.1):
        #     for j in np.arange(-6,6,0.1):
        #         tmp = self.predict([i, j])
        #         if tmp[0]>0.5:
        #             plt.scatter(i,j,c='r',s=5,alpha=0.4,marker='o')
        #         elif tmp[0]<0.5:
        #             plt.scatter(i,j,c='c',s=5,alpha=0.4,marker='o')
        # plt.show()

        test_cases=[]
        test_labels=[]
        test_set=open('../data/test.txt', 'r')
        lines = train_set.readlines()
        result=open('../data/predit.txt', 'w')
        for line in lines:
            sample = line.replace('\t\n', '').split('\t')
            vector = []
            label = []
            for j in range(len(sample)-1):
                vector.append(float(sample[j]))
            label.append(float(sample[-1]))
            test_cases.append(vector)
            test_labels.append(label)

        for i in test_cases:
            result=self.predict(i)
            result.write(str(test_labels[i])+'\t'+str(result)+'\n')



        # cases = [
        #     [0, 0],
        #     [0, 1],
        #     [1, 0],
        #     [1, 1],
        # ]
        # labels = [[0], [1], [1], [0]]
        # self.setup(2, 5, 1)
        # self.train(cases, labels, 10000, 0.05, 0.1)
        # for case in cases:
        #     print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()