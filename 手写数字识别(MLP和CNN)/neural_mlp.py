from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

# 1. 获取数据 & 初步分析
#   获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#   初步分析
#       数据的维度信息
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
#       数据的图像信息
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(x_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig
plt.show()

# 2. 数据预处理 & 准备训练数据和测试数据
#       shape:
#               (60000, 28, 28)  三维数组：60000个28*28的二维矩阵
#               (60000,)
#       reshape:
#               修改数组（矩阵）形状  行和列
#       [a:b, None]：
#               (a,b)是选取切片  |  None== numpy.newaxis 是创建新轴（改变数组维度），这里相当于增加 “ 列 ”
#               y_train一开始的shape是(60000,), 这里应该是 (60000,)  --> (60000,784)
#       astype:
#               转换数组的数据类型
#   将三维向量转换成二维向量
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
#   归一化
x_train = x_train / 255
x_test = x_test / 255
#   编码
#      将类别0~9用onehot编码，目的是配合softmax
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 2. 构建网络层
model = Sequential()
#   输入层
model.add(Dense(512, input_shape=(784,)))  # 28*28=784 把二维矩阵转换成一维矩阵
model.add(Activation('relu'))
model.add(Dropout(0.2))
#   隐藏层
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#   输出层
#       10 对应最终的分类数目0~9
#       使用softmax而不是tanh，避免有的分类概率为0
model.add(Dense(10))
model.add(Activation('softmax'))
# 输出模型的整体信息
model.summary()

# 3. 设置网络优化器
#       这里使用 随机梯度下降
#       损失函数使用交叉熵
#       SGD参数：
#               lr 学习率（大于0的浮点数） momennum 动量参数（大于0的浮点数）
#               decay 每次更新后的学习率衰减值（大于0的浮点数）
#               nesteroy 是否使用Nesteroy动量
#               metrics 在训练和测试期间的模型评估标准
sgd = SGD(learning_rate=0.01, decay=0.000001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 4. 模型训练
#    fit的参数：
#           batch_size：对总的样本数进行分组，每组包含的样本数量
#           epochs ：训练次数
#           shuffle：是否把数据随机打乱之后再进行训练
#           validation_split：拿出百分之多少用来做交叉验证
#           verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
train_history = model.fit(x_train, y_train, batch_size=128, epochs=50, shuffle=True, verbose=2, validation_split=0.3)

# 5. 模型测试
model_predict = model.predict(x_test, batch_size=128, verbose=1)
predict_max = np.argmax(model_predict, axis=1)  # axis: 0 列  1 行  None 全部
test_max = np.argmax(y_test, axis=1)

# 6. 模型分析
#   数值指标
flags = np.equal(predict_max, test_max)
sucess_num = np.sum(flags)
print("accuracy: %f", sucess_num / len(flags))
print("end")

#   训练过程的统计图
# accuracy 是使用训练集计算准确度
# val_accuracy 是使用验证数据集计算准确度
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('Train History')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
