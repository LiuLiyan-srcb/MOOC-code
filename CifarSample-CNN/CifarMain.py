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
            all_data.append(data)
            all_labels.append(labels)
        # from list to array
        self._data = np.vstack(all_data)
        # normalize image data to [-1,1]
        self._data = self._data / 127.5 -1
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

#构建tensorflow计算图
x = tf.placeholder(tf.float32, [None, 3072])
#[None] e.g. [0,6,5,3....]
y = tf.placeholder(tf.int64,[None])


x_image = tf.reshape(x, [-1,3,32,32])
# 把后两个通道移到前面 成为32x32x3
x_image = tf.transpose(x_image, perm=[0,2,3,1])

# feature map, or 输出图像, or 神经元图
conv1 = tf.layers.conv2d(x_image,
                         32, #output channel number
                         (3,3),#kernel size
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv1')
# output - 16x16
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2,2), # kernel size
                                   (2,2), # stride
                                   name = 'pool1' )

conv2 = tf.layers.conv2d(pooling1,
                         32, #output channel number
                         (3,3),#kernel size
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv2')
# output - 8x8
pooling2 = tf.layers.max_pooling2d(conv2,
                                   (2,2), # kernel size
                                   (2,2), # stride
                                   name = 'pool2' )

conv3 = tf.layers.conv2d(pooling2,
                         32, #output channel number
                         (3,3),#kernel size
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv3')

# output - 4x4x32
pooling3 = tf.layers.max_pooling2d(conv3,
                                   (2,2), # kernel size
                                   (2,2), # stride
                                   name = 'pool3' )

# [None, 4x4x32]
flatten = tf.layers.flatten(pooling3)
# 全连接层,对应到10个类
y_ =tf.layers.dense(flatten, 10)

# hidden_1 = tf.layers.dense(x,100, activation=tf.nn.relu)
# hidden_2 = tf.layers.dense(hidden_1, 100, activation=tf.nn.relu)
# hidden_3 = tf.layers.dense(hidden_2, 50, activation=tf.nn.relu)
# y_ = tf.layers.dense(hidden_3,10)
# this is for multi-classificatoin
# mean square loss


# cross entropy loss
# y_ -> softmax
# y -> one_hot
# loss = ylog(y_)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


# this is for bi-classification
#[None,1]
# p_y_1 = tf.nn.sigmoid(y_)
# #[None,1]
# y_reshaped = tf.reshape(y, (-1,1))
# y_reshaped_float = tf.cast(y_reshaped, tf.float32)
#
# loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))


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