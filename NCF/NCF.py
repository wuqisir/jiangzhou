import numpy as np
import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import sys
import Metrics

class NCF(object):
    def __init__(self, embed_size, user_size, item_size, lr,
                 optim, initializer, loss_func, activation_func,
                 regularizer_rate, iterator, topk, dropout, is_training):

        self.embed_size = embed_size
        self.user_size = user_size
        self.item_size = item_size
        self.lr = lr
        self.initializer = initializer
        self.loss_func = loss_func
        self.activation_func = activation_func
        self.regularizer_rate = regularizer_rate
        self.optim = optim
        self.topk = topk
        self.dropout = dropout
        self.is_training = is_training
        self.iterator = iterator

    def get_data(self):
        sample = self.iterator.get_next()
        self.user = sample['user']
        self.item = sample['item']
        self.label = tf.cast(sample['label'], tf.float32)

    def inference(self):
        # 初始化参数设置
        # self.regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate) #正则化L2全连接神经网络
        # self.regularizer = tf.layers.l2_regularizer(self.regularizer_rate)  # 正则化L2全连接神经网络
        # self.regularizer = tf.nn.l2_loss(self.regularizer_rate)
        self.regularizer = tf.keras.regularizers.l2(self.regularizer_rate)  # 使用这个方法在compat下调用l2正则

        if self.initializer == 'Normal':
            self.initializer = tf.truncated_normal_initializer(stddev=0.01)  # 截断正态分布
        elif self.initializer == 'Xavier_Normal':
            self.initializer = tf.contrib.layers.xavier_initializer()  # 用来使得每一层输出的方差应该尽量相等
        elif self.initializer == 'original':
            self.initializer = tf.random_normal_initializer(stddev=1.5)
        elif self.initializer == 'uniform':
            self.initializer = tf.random_uniform_initializer(-1., 1.)
        else:
            self.initializer = tf.glorot_uniform_initializer()  # 均匀分布

        if self.activation_func == 'ReLU':
            self.activation_func = tf.nn.relu
        elif self.activation_func == 'Leaky_ReLU':
            self.activation_func = tf.nn.leaky_relu
        elif self.activation_func == 'ELU':
            self.activation_func = tf.nn.elu

        if self.loss_func == 'cross_entropy':
            self.loss_func = tf.nn.sigmoid_cross_entropy_with_logits

        if self.optim == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(self.lr, name='SGD')
        elif self.optim == 'RMSProp':
            self.optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9, momentum=0.0, name='RMSProp')
        elif self.optim == 'Adam':
            self.optim = tf.train.AdamOptimizer(self.lr, name='Adam')  # 自适应矩估计是一种基于一阶梯度的随机目标函数优化算法

    def create_model(self):
        with tf.name_scope('input'):
            self.user_onehot = tf.one_hot(self.user, self.user_size, name='user_onehot')
            self.item_onehot = tf.one_hot(self.item, self.item_size, name='item_onehot')

        with tf.name_scope('embed'):
            self.user_embed_GMF = tf.layers.dense(inputs=self.user_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_GMF')

            self.item_embed_GMF = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_GMF')

            self.user_embed_MLP = tf.layers.dense(inputs=self.user_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_MLP')

            self.item_embed_MLP = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_MLP')

        with tf.name_scope('GMF'):
            self.GMF = tf.multiply(self.user_embed_GMF, self.item_embed_GMF, name='GMF')

        with tf.name_scope('MLP'):
            self.interaction = tf.concat([self.user_embed_MLP, self.item_embed_MLP],
                                         axis=-1, name='interaction')

            self.layer1_MLP = tf.layers.dense(inputs=self.interaction,
                                              units=self.embed_size * 2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer1_MLP')
            self.layer1_MLP = tf.layers.dropout(self.layer1_MLP, rate=self.dropout)

            self.layer2_MLP = tf.layers.dense(inputs=self.layer1_MLP,
                                              units=self.embed_size,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer2_MLP')
            self.layer2_MLP = tf.layers.dropout(self.layer2_MLP, rate=self.dropout)

            self.layer3_MLP = tf.layers.dense(inputs=self.layer2_MLP,
                                              units=self.embed_size // 2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer3_MLP')
            self.layer3_MLP = tf.layers.dropout(self.layer3_MLP, rate=self.dropout)

        with tf.name_scope('concatenation'):
            self.concatenation = tf.concat([self.GMF, self.layer3_MLP], axis=-1, name='concatenation')

            self.logits = tf.layers.dense(inputs=self.concatenation,
                                          units=1,
                                          activation=None,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          name='prediction')
            self.logits_dense = tf.reshape(self.logits, [-1])  # [-1]:不指定维度，函数自动计算

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.label, logits=self.logits_dense, name='loss'))  # 降维计算tensor平均值

        with tf.name_scope('optimzation'):
            self.optimzer = self.optim.minimize(self.loss)

    # def eval(self):
    #     with tf.name_scope('evaluation'):
    #         self.item_replica = self.item
    #         _, self.indice = tf.nn.top_k(tf.sigmoid(self.logits_dense), self.topk)

    def eval(self):
        with tf.name_scope('evaluation'):
            self.item_replica = self.item
            self.result = tf.sigmoid(self.logits_dense)

    # def summary(self):
    #     self.writer = tf.summary.FileWriter('graphs/', tf.get_default_graph())
    #     with tf.name_scope('summaries'):
    #         tf.summary.scalar('loss', self.loss)
    #         tf.summary.histogram('histogram loss',self.loss)
    #         self.summary_op = tf.summary.merge_all()

    def build(self):
        self.get_data()
        self.inference()
        self.create_model()
        self.eval()
        # self.summary()
        # self.saver = tf.train.Saver(tf.global_variables())

    # def step(self, session, step):
    #     if self.is_training:
    #         loss, optim, summaries = session.run(
    #             [self.loss, self.optimzer, self.summary_op])
    #         self.writer.add_summary(summaries, global_step=step)
    #     else:
    #         indice, item = session.run([self.indice, self.item_replica])
    #         prediction = np.take(item, indice)
    #
    #         return prediction, item

    def step(self, session, step):
        if self.is_training:
            loss, optim = session.run(
                [self.loss, self.optimzer])
            # self.writer.add_summary(summaries, global_step=step)
        else:
            result = session.run(self.result)
            # print(label,prediction)
            # print(len(prediction))
            return result
