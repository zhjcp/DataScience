import numpy as np
import torchvision


def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


# 标签one-hot处理
def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid的一阶导数
def Dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NN(object):
    def __init__(self, l0, l1, l2, l3, batch_size=6):
        self.lr = 0.4  # 学习率
        self.batch_size = batch_size
        self.W1 = np.random.randn(l0, l1) * 0.01  # 初始化
        self.b1 = np.random.randn(l1) * 0.01
        self.W2 = np.random.randn(l1, l2) * 0.01
        self.b2 = np.random.randn(l2) * 0.01
        self.W3 = np.random.randn(l2, l3) * 0.01
        self.b3 = np.random.randn(l3) * 0.01

    # 前向传播
    def forward(self, X, y):
        self.X = X  # m x 784
        self.z1 = np.dot(X, self.W1) + self.b1  # m x 200, 等于中间层层数
        self.a1 = sigmoid(self.z1)  # m x 200

        # self.z2 = np.dot(self.a1, self.W2) + self.b2         # m x 30
        # self.a2 = sigmoid(self.z2)                           # m x 30

        self.z2 = np.dot(self.a1, self.W2) + self.b2  # m x 10
        self.a2 = sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)

        loss = np.sum((self.a3 - y) * (self.a3 - y)) / 6  # 均方差

        self.d3 = (self.a3 - y) * Dsigmoid(self.z3)  # 用于反向传播
        return loss, self.a3

    # 反向传播
    def backward(self):
        dW3 = np.dot(self.a2.T, self.d3) / self.batch_size
        db3 = np.sum(self.d3, axis=0) / self.batch_size

        d2 = np.dot(self.d3, self.W3.T) * Dsigmoid(self.z2)  # 用于反向传播
        dW2 = np.dot(self.a1.T, d2) / self.batch_size
        db2 = np.sum(d2, axis=0) / self.batch_size

        d1 = np.dot(d2, self.W2.T) * Dsigmoid(self.z1)
        dW1 = np.dot(self.X.T, d1) / self.batch_size  # 784x 200
        db1 = np.sum(d1, axis=0) / self.batch_size  # 200

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


def train():
    nn = NN(784, 200, 30, 10)

    for epoch in range(10):
        for i in range(0, 60000, nn.batch_size):
            X = train_data.data[i:i + nn.batch_size]
            Y = train_data.targets[i:i + nn.batch_size]
            loss, _ = nn.forward(X, Y)
            print("epocj:", epoch, "-", i, ":", "{:.3f}".format(loss))
            nn.backward()
        np.savez("data.npz", w1=nn.W1, b1=nn.b1, w2=nn.W2, b2=nn.b2, w3=nn.W3, b3=nn.b3)


def test():
    r = np.load("data.npz")
    nn = NN(784, 200, 30, 10)
    nn.W1 = r["w1"]
    nn.b1 = r["b1"]
    nn.W2 = r["w2"]
    nn.b2 = r["b2"]
    nn.W3 = r["w3"]
    nn.b3 = r["b3"]

    _, result = nn.forward(test_data.data, test_data.targets2)
    result = np.argmax(result, axis=1)
    precison = np.sum(result == test_data.targets) / 10000
    print("Precison:", precison)


if __name__ == '__main__':
    # Mnist手写数字集
    train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True)
    test_data = torchvision.datasets.MNIST(root='data/', train=False)
    train_data.data = train_data.data.numpy()  # [60000,28,28]
    train_data.targets = train_data.targets.numpy()  # [60000]
    test_data.data = test_data.data.numpy()  # [10000,28,28]
    test_data.targets = test_data.targets.numpy()  # [10000]

    # 输入向量处理
    train_data.data = train_data.data.reshape(60000, 28 * 28) / 255.  # (60000, 784)

    test_data.data = test_data.data.reshape(10000, 28 * 28) / 255.

    # 标签one-hot处理
    train_data.targets = onehot(train_data.targets, 60000)  # (60000, 10)
    test_data.targets2 = onehot(test_data.targets, 10000)  # 用于前向传播

    train()
    test()