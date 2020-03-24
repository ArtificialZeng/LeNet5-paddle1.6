# 导入需要的包
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC

# 定义 LeNet 网络结构
class LeNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes=1):
        super(LeNet, self).__init__(name_scope)
        name_scope = self.full_name()
        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(name_scope, num_filters=6, filter_size=5, act='sigmoid')
        self.pool1 = Pool2D(name_scope, pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(name_scope, num_filters=16, filter_size=5, act='sigmoid')
        self.pool2 = Pool2D(name_scope, pool_size=2, pool_stride=2, pool_type='max')
        # 创建第3个卷积层
        self.conv3 = Conv2D(name_scope, num_filters=120, filter_size=4, act='sigmoid')
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分裂标签的类别数
        self.fc1 = FC(name_scope, size=64, act='sigmoid')
        self.fc2 = FC(name_scope, size=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 输入数据形状是 [N, 3, H, W]
# 这里用np.random创建一个随机数组作为输入数据
x = np.random.randn(*[3,3,28,28])
x = x.astype('float32')
with fluid.dygraph.guard():
    # 创建LeNet类的实例，指定模型名称和分类的类别数目
    m = LeNet('LeNet', num_classes=10)
    # 通过调用LeNet从基类继承的sublayers()函数，
    # 查看LeNet中所包含的子层
    print(m.sublayers())
    x = fluid.dygraph.to_variable(x)
    for item in m.sublayers():
        # item是LeNet类中的一个子层
        # 查看经过子层之后的输出数据形状
        x = item(x)
        if len(item.parameters())==2:
            # 查看卷积和全连接层的数据和参数的形状，
            # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
            print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
        else:
            # 池化层没有参数
            print(item.full_name(), x.shape)
      
  
# -*- coding: utf-8 -*-

# LeNet 识别手写数字

import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np

# 定义训练过程
def train(model):
    print('start training ... ')
    model.train()
    epoch_num = 5
    opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
    # 使用Paddle自带的数据读取器
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=10)
    valid_loader = paddle.batch(paddle.dataset.mnist.test(), batch_size=10)
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            # 调整输入数据形状和类型
            x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
            y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
            # 将numpy.ndarray转化成Tensor
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss = fluid.layers.softmax_with_cross_entropy(logits, label)
            avg_loss = fluid.layers.mean(loss)
            if batch_id % 1000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            # 调整输入数据形状和类型
            x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
            y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
            # 将numpy.ndarray转化成Tensor
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 计算模型输出
            logits = model(img)
            pred = fluid.layers.softmax(logits)
            # 计算损失函数
            loss = fluid.layers.softmax_with_cross_entropy(logits, label)
            acc = fluid.layers.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

    # 保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')


if __name__ == '__main__':
    # 创建模型
    with fluid.dygraph.guard():
        model = LeNet("LeNet", num_classes=10)
        #启动训练过程
        train(model)
