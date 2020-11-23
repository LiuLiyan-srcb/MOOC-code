import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import _pickle as cPickle

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

#read data from data file
def load_data(filename):
    with open(filename, 'rb') as f:
    #add 'encoding = 'bytes' to solve 'UnicodeDecodeError'
    #data.keys() are b'data', b'labels'
        data = cPickle.load(f, encoding='bytes')
        # print (type(data))
        # print(data.keys())
        # print(data[b'labels'])
        return data[b'data'], data[b'labels']

class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            for item, label in zip(data, labels):
                if  label in [0,1]:
                    all_data.append(item)
                    all_labels.append(label)
        # from list to array
        self._data = np.vstack(all_data)
        # normalize image data to [-1,1]
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        # print( self._data.shape )
        # print( self._labels.shape )

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()


    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch"""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

# batch_data, batch_labels = train_data.next_batch(10)
# print (batch_data)
# print (batch_labels)

# ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。
# ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习
# residual connection implementation
# x: input
# output_channel:
def residual_block(x, output_channel):
    input_channel = x.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2,2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1,1)
    else:
        print("exception: input channel cannot match output channel")

    conv1 = tf.layers.conv2d(x,
                              output_channel,
                              (3,3),
                              strides = strides,
                              padding = 'same',
                              activation=tf.nn.relu,
                              name = 'cconv1')

    conv2 = tf.layers.conv2d(conv1,
                             output_channel,
                             (3, 3),
                             strides=(1,1),
                             padding='same',
                             activation=tf.nn.relu,
                             name='cconv2')
    if increase_dim:
        #[None, image_width, image_height, channel] -> [,,,channel*2]
        pooled_x = tf.layers.average_pooling2d(x,
                                               (2,2),
                                               (2,2),
                                               padding='valid')
        padded_x = tf.pad(pooled_x,
                          [[0,0],
                          [0,0],
                          [0,0],
                          [input_channel // 2, input_channel//2]])
    else:
        padded_x = x

    # conv2 即为残差，padded_x 即为恒等映射，只是要考虑到某些层需要把x的输入尺寸降维为原来的一半，再与残差相加
    output_x = conv2 + padded_x

    return output_x

## resudula network implementation
# x - input
# num_residual_blocks: 每一组内残差块的数目 eg[3,4,6,3] 表示有4个stage，每个stage上残差连接的数目3，4，6，3
# num_subsampling: 下采样的数目 与 num_residual_blocks列表长度相同，按上面的例子，其长度即为4, 实现时可省略这个输入参数
def res_net(x,
            num_residual_blocks,
            num_filter_base,
            class_num):

    num_subsampling = len(num_residual_blocks)
    layers = []
    # x:[None, width, height, channel] -> [width, height, channel]
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3,3),
                                 strides = (1,1),
                                 padding = 'same',
                                 activation = tf.nn.relu,
                                 name = 'conv0')
        layers.append(conv0)

    # num_subsampling = 4, sample_id = [0,1,2,3]
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope("conv%d_%d" % (sample_id, i)):
                conv = residual_block(layers[-1],
                                      num_filter_base * (2** sample_id))

                layers.append(conv)

    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] \
         == [input_size[0] / multiplier,
             input_size[1] / multiplier,
             num_filter_base * multiplier]

    with tf.variable_scope('fc'):
        # layer[-1].shape: [None, width, height, channel]
        # kernel size:  image_width, image_height
        # 将一张图像变为一个均值
        global_pool = tf.reduce_mean(layers[-1],[1,2])
        # FC layer
        logits = tf.layers.dense(global_pool, class_num)
        layers.append(logits)

    return layers[-1]

#构建tensorflow计算图
x = tf.placeholder(tf.float32, [None, 3072])

#[None]
y = tf.placeholder(tf.int64,[None])

x_image = tf.reshape(x, [-1,3,32,32])
# 把后两个通道移到前面 成为32x32x3
x_image = tf.transpose(x_image, perm=[0,2,3,1])

# feature map, or 输出图像, or 神经元图
# ResNet 实战

y_ = res_net(x_image, [2,3,2], 32, 10)

# cross entropy loss
# y_ -> softmax
# y -> one_hot
# loss = ylog(y_)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

#indices
predict = tf.argmax(y_, 1)
#[1,0,1,1,0,0,1,....]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))


with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, accu_val, _ = sess.run([loss, accuracy,train_op],
                                         feed_dict={
                                             x:batch_data,
                                             y:batch_labels})
        if (i+1) % 500 == 0:
            print( '[Train] Step: %d, loss: %4.5f, acc: %4.5f' \
            % (i, loss_val, accu_val) )

        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_lables = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],
                                        feed_dict={x: test_batch_data,
                                                   y: test_batch_lables})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test] Step: %d, acc: %4.5f' % (i+1, test_acc))