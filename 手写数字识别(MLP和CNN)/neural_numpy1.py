from tensorflow.keras.datasets import mnist
import numpy as np
import random
import tensorflow


def sigmoid(inX):
    from numpy import exp
    return 1.0 / (1 + exp(-inX))


def dsigmoid(z):
    return sigmoid(z)(1 - sigmoid(z))


class MLP:

    def __init__(self, sizes):
        """
        :param size: [784,30,10]
        """
        self.size = sizes
        self.num_layers = len(sizes) - 1
        # size[784,30,10]
        # w:[输出，输入]
        # b:[输出]
        self.weight = [np.random.randn(ch2, ch1)
                       for ch1, ch2 in zip(sizes[:-1], sizes[1:])]
        # [784,30],[30,10]  z=wxx+b [30,1]
        self.bias = [np.random.rand(s, 1) for s in sizes[1:]]

    def forward(self, x):
        """

        :param x: [784,1]
        :return: [10]
        """
        for b, w in zip(self.bias, self.weight):
            # [30,784]@[784,1]->[30,1]+[30,1]=[30,1]
            z = np.dot(w, x) + b
            x = sigmoid(z)

        return x

    def backprop(self, x, y):
        """

        :param x: [784,1]
        :param y: [10,1]
        :return:
        """
        x = x.reshape(784, 1)

        nabla_w = [np.zeros(w.shape) for w in self.weight]
        nabla_b = [np.zeros(b.shape) for b in self.bias]

        # 1.forward
        # 保存每一层的激活参数
        activations = [x]
        # 保存每一层的中间结果z
        zs = []
        activation = x
        for b, w in zip(self.bias, self.weight):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)

            zs.append(z)
            activations.append(activation)

        loss = np.power((activations[-1] - y), 2).sum()
        # 2.backward
        # 2.1计算输出层的梯度
        # [10,1] [10,1] ->[10,1]
        delta = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)
        nabla_b[-1] = delta
        # [10,1]@[1,30] -> [10,30]
        # activation:[30,1]
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # 2.2 compute hidden grendient
        for l in range(2, self.num_layers + 1):
            l = -l

            z = zs[l]
            a = activations[l]

            # delta_j
            # [10,30]T @ [10,1]  =>  [30,10] @ [10,1] =>[30,1] *[30,1] =>[30,1]
            delta = np.dot(self.weight[l + 1].T, delta) * a * (1 - a)

            nabla_b[l] = delta
            # [30,1] @ [784,1]T => [30,784]
            nabla_w[l] = np.dot(delta, activations[l - 1].T)

        return nabla_w, nabla_b, loss

    def train(self, training_data, epoches, batchsz, lr, test_data):
        """
        :param training_data: list of (x,y)
        :param epoches: 1000
        :param batchsz: 10
        :param lr: 0.1
        :param test_data: list of (x,y)
        :return:
        """
        n = len(training_data)
        for j in range(epoches):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + batchsz]
                for k in range(0, n, batchsz)]

            # for every batch in current batch
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, lr)
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test), loss)
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, lr):
        """

        :param batch: list of (x,y)
        :param lr: 0.01
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        loss = 0
        # for every sample in current batch
        for x, y in batch:
            # list of every w/b gradient
            # [w1,w2,w3]
            nabla_w_, nabla_b_, loss_ = self.backprop(x, y)
            nabla_w = [accu + cur for accu, cur in zip(nabla_w, nabla_w_)]
            nabla_b = [accu + cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_
        nabla_w = [w / len(batch) for w in nabla_w]
        nabla_b = [b / len(batch) for b in nabla_b]
        loss = loss / len(batch)
        # w = w - lr * nabla_w
        self.weight = [w - lr * nabla for w, nabla in zip(self.weight, nabla_w)]
        self.bias = [b - lr * nabla for b, nabla in zip(self.bias, nabla_b)]

        return loss

    def evaluate(self, test_data):
        """

        :param test_data: list of (x,y)
        :return:
        """
        result = [(np.argmax(self.forward(x.reshape([784, 1]))), y)
                  for x, y in test_data]

        correct = sum(int(pred == y) for pred, y, in result)

        return correct


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T


def execute_neural():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_data = []
    train_x = train_x.reshape([60000, 784])
    for i in range(train_x.shape[0]):
        # print(convert_to_one_hot(train_y[i],10).shape)
        train_data.append([train_x[i] / 255, convert_to_one_hot(train_y[i], 10)])

    test_data = []
    test_x = test_x.reshape([10000, 784])
    for i in range(10000):
        test_data.append([test_x[i] / 255, test_y[i]])

    net = MLP([784, 30, 10])
    net.train(train_data, 20, 10, 0.1, test_data=test_data)


execute_neural()
