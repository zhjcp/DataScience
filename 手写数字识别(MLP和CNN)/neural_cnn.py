from keras import models, layers
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# 搭建LeNet网络
#   卷积层：
#       卷积层有比较强的维度提升效果，如果层数比较多，输出维度会很恐怖。
#
#   池化层：
#       池化层可以降维，控制卷积层的输出维度，进而支持更深的网络和更强的表征能力。并且能够在一定程度上控制过拟合。
#
#   Flatten：
#       Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小
#
#   全连接层：
#       全连接层的作用主要就是实现分类
#       如果说卷积层、池化层和激活函数层等操作是将原始数据映射到隐层特征空间的话，全连接层则起到将学到的“分布式特征表示”映射到样本标记空间的作用
def LeNet():
    network = models.Sequential()
    #   卷积层
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    #       平均池化
    network.add(layers.AveragePooling2D((2, 2)))
    #   卷积层
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    #       平均池化
    network.add(layers.AveragePooling2D((2, 2)))
    #   卷积层
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    #       展平
    network.add(layers.Flatten())
    #   全连接层
    network.add(layers.Dense(84, activation='relu'))
    #   输出层
    network.add(layers.Dense(10, activation='softmax'))
    return network


model = LeNet()
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#   维度、数据类型、归一化
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
#   one-hot编码
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
train_history = model.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)

# 统计图
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
