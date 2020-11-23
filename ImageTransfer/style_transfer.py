#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import time

#R,G,B channel mean value
VGG_MEAN = [103.939, 116.779, 123.68]

class VGGNet:
    """Builds VGG-16 net structure,
       load parameters from pre-train models.
    """
    def __init__(self, data_dict):
        self.data_dict = data_dict
    
    def get_conv_filter(self, name):
        # [0] w value
        # [1] b value
        return tf.constant(self.data_dict[name][0], name='conv')
    
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')
    
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')
    
    def conv_layer(self, x, name):
        """Builds convolution layer."""
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            h = tf.nn.conv2d(x, conv_w, [1,1,1,1], padding='SAME')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)
            return h
    
    
    def pooling_layer(self, x, name):
        """Builds pooling layer."""
        return tf.nn.max_pool(x,
                              # [batch_size, image_width, image_height, channel]
                              ksize = [1,2,2,1],
                              strides = [1,2,2,1],
                              padding = 'SAME',
                              name = name)
    
    def fc_layer(self, x, name, activation=tf.nn.relu):
        """Builds fully-connected layer."""
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            if activation is None:
                return h
            else:
                return activation(h)
    
    def flatten_layer(self, x, name):
        """Builds flatten layer."""
        with tf.name_scope(name):
            # [batch_size, image_width, image_height, channel]
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])
            return x
    
    def build(self, x_rgb):
        """Build VGG16 network structure.
        Parameters:
        - x_rgb: [1, 224, 224, 3]
        batch size 1
        image size 224x224, channel 3
        """

        # to calculate time for building model
        start_time = time.time()
        print ('building model ...')

        #axis 第几个维度
        r, g, b = tf.split(x_rgb, [1,1,1], axis=3)
        # 减去VGG均值， 排列顺序 b,g,r
        x_bgr = tf.concat(
            [b - VGG_MEAN[0],
             g - VGG_MEAN[1],
             r - VGG_MEAN[2]],
            axis = 3)
        
        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 开始构建VGG网络
        # 注意，输入并不是最原始的图像输入，而是减去均值后的输入
        # 每经过一个Pooling层，叫做经过一个stage （本课程的叫法）
        # VGG16： 13个卷积层 + 3个全连接层
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')
        
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')
        
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')
        
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')
        
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')
        

        # 进行图像风格转换的时候，只用到卷积层的feature，用不到全连接层，可以注释掉
        # VGG 的网络构建，大部分时间花在全连接层上
        self.flatten5 = self.flatten_layer(self.pool5, 'flatten')
        self.fc6 = self.fc_layer(self.flatten5, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        self.prob = tf.nn.softmax(self.fc8, name='prob')


        
        print ('building model finished: %4ds' % (time.time() - start_time))


# In[ ]:


vgg16_npy_path = './vgg16.npy'
content_img_path = './gugong.jpg'
style_img_path = './star.jpg'

num_steps = 100
learning_rate = 10

# 在生成转换内容时的两个加权系数
lambda_c = 0.1
lambda_s = 500

output_dir = './run_style_transfer'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# In[ ]:


def initial_result(shape, mean, stddev):
    # 截断正态分布
    initial = tf.truncated_normal(shape, mean = mean, stddev = stddev)
    return tf.Variable(initial)

def read_img(img_name):
    img = Image.open(img_name)
    np_img = np.array(img) # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    return np_img

def gram_matrix(x):
    """Calulates gram matrix
    Args:
    - x: feaures extracted from VGG Net. shape: [1, width, height, ch]
    """
    b, w, h, ch = x.get_shape().as_list()
    # 把w,h合并到一个维度
    features = tf.reshape(x, [b, h*w, ch]) # [ch, ch] -> (i, j)
    # 在每两个channel之间计算特征相似度
    # 首先进行矩阵转置，然后与原矩阵进行乘法，得到的就是两个channel之间‘相乘’的结果，也就是相似度
    # [h*w, ch] matrix -> [ch, h*w] * [h*w, ch] -> [ch, ch]
    # adjoint_a = True 表示要进行转置
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
    return gram
    


content_val = read_img(content_img_path)
style_val = read_img(style_img_path)

# 用于输入到VGG中
content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

# 随机初始化一个result图像
result = initial_result((1, 224, 224, 3), 127.5, 20)


# add allow_pickle=True encoding='latin1'
data_dict = np.load('vgg16.npy', allow_pickle=True, encoding='latin1').item()
# data_dict = np.load(vgg16_npy_path).item() 这句会报错  Object arrays cannot be loaded when allow_pickle=False

# 为源图像，风格图像，结果图像分别创建网络，提取图像特征
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)

vgg_for_content.build(content)
vgg_for_style.build(style)
vgg_for_result.build(result)

# VGG中的不同卷积层都可以用于特征提取
# content 用低层特征较好
# 可以同时计算多层，但是需要花费更多的时间
content_features = [
    vgg_for_content.conv1_2,
    # vgg_for_content.conv2_2,
    # vgg_for_content.conv3_3,
    # vgg_for_content.conv4_3,
    # vgg_for_content.conv5_3
]

# 结果图像内容的特征需要与内容图像一致，例如，都使用conv1_2
result_content_features = [
    vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    # vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]

# feature map
# feature_size, [1, width, height, channel]
# 风格特征更偏向于高层特征
style_features = [
    # vgg_for_style.conv1_2,
    # vgg_for_style.conv2_2,
    # vgg_for_style.conv3_3,
    vgg_for_style.conv4_3,
    # vgg_for_style.conv5_3
]
# 假设有ｎ个ｃｈａｎｎｅｌ，每个ｃｈａｎｎｅｌ上得到ｗｉｄｔｈｘｈｅｉｇｈｔ的特征图。
# 每两个特征图之间，计算相似性（余弦距离），得到gram矩阵
# 风格图像的每一个风格特征计算gram矩阵
style_gram = [gram_matrix(feature) for feature in style_features]

# 结果图像风格的特征与style相一致
result_style_features = [
    # vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]
result_style_gram =  [gram_matrix(feature) for feature in result_style_features]

# 计算内容损失， 风格损失
# 内容特征可以来自多层，因此损失要对多层进行累加
content_loss = tf.zeros(1, tf.float32)
# zip: 把两个数组一一绑定在一起
# zip: [1, 2], [3, 4], zip([1,2], [3,4]) -> [(1, 3), (2, 4)]
# shape: [1, width, height, channel]
for c, c_ in zip(content_features, result_content_features):
    # 平方差后取平均 （在后三个通道1-width，2-height，3-channel上都要取平均）
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])

#风格图像提取的特征与结果图像的风格特征之间进行损失函数计算
style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])

# 总的损失函数，等于两个损失函数的加权
loss = content_loss * lambda_c + style_loss * lambda_s
# 为损失函数计算梯度
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[ ]:


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        loss_value, content_loss_value, style_loss_value, _  = sess.run([loss, content_loss, style_loss, train_op],
                     feed_dict = {
                         content: content_val,
                         style: style_val,
                     })
        print ('step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f'
               % (step+1,
               loss_value[0],
               content_loss_value[0],
               style_loss_value[0]))
        result_img_path = os.path.join(
            output_dir, 'result-%05d.jpg' % (step+1))
        result_val = result.eval(sess)[0]
        result_val = np.clip(result_val, 0, 255)
        img_arr = np.asarray(result_val, np.uint8)
        img = Image.fromarray(img_arr)
        img.save(result_img_path)


# In[ ]:




