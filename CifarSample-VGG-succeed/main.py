import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import _pickle as cPickle

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

#tensorboard
#指定面板图上要显示的变量
#训练过程中将变量计算出来，输出到文件中
#文件解析 ./tensorboard --logdir = dir.

#activition: relu, sigmoid, tanh
#weight initializer: he, xavier, normal
#optimizer: Adam, Momentum, Gradient Descent

#fine tune
# 1. save models (一般由third_party提供，在自己的代码中使用)
# 2. restore models
# 3. keep some layers fixed


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
        # 不做图像增强的时候用这个语句
        #self._data = self._data / 127.5 - 1

        #为了进行图像增强，此处先不对图像进行归一化处理
        self._data = self._data

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

#[None]
y = tf.placeholder(tf.int64,[None])

is_training = tf.placeholder(tf.bool, [])

# 不做图像增强： x_image是归一化到【-1，1】的数据
# 做图像增强： x_image就是图像数据【0，255】
x_image = tf.reshape(x, [-1,3,32,32])
# 把后两个通道移到前面 成为32x32x3
x_image = tf.transpose(x_image, perm=[0,2,3,1])

#图像增强
data_aug_1 = tf.image.random_flip_left_right(x_image)
data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=63)
data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=1.8)

#封装conv_2d
# def conv_wrapper(inputs, name, output_channel=32, kernel_size=(3,3),activation=tf.nn.relu,padding='same'):
#     return  tf.layers.conv2d(inputs,
#                              output_channel,
#                              kernel_size,
#                              padding = padding,
#                              activation = activation,
#                              name = name)
#加入batch normalization
#without bn: conv -> activation
#with bn: conv -> bn -> activation
# is_training: training stage - True, test stage - False

def conv_wrapper(inputs,
                 name,
                 is_training,
                 output_channel = 32,
                 kernel_size = (3,3),
                 activation = tf.nn.relu,
                 padding = 'same',
                 ):
    with tf.name_scope(name):
        conv2d =  tf.layers.conv2d(inputs,
                                 output_channel,
                                 kernel_size,
                                 padding = padding,
                                 activation = None,
                                 name = name + '/conv1')
        bn = tf.layers.batch_normalization(conv2d,
                                           training = is_training)
        return activation(bn)

#封装pooling层
def pooling_wrapper(inputs, name):
    return tf.layers.max_pooling2d(inputs,
                                   (2,2),
                                   (2,2),
                                   name = name)

#封装conv方法，以便使用不同的activation
# 从2个卷积层 + 1个pooling层，加深到3个卷积层
def convnet(inputs, is_training, activation, kernel_initializer):
    # feature map, or 输出图像, or 神经元图
    # 将普通的CNN改造为VGG思想的网络
    # 每个卷积层后各增加一个卷积层，加深层次
    # 每两个卷积层 + 一个pooling 层
    conv1_1 = conv_wrapper(inputs,'conv1_1',is_training)
    conv1_2 = conv_wrapper(conv1_1, 'conv1_2',is_training)
    conv1_3 = conv_wrapper(conv1_2, 'conv1_3',is_training)
    pooling1 = pooling_wrapper(conv1_3, 'pool1')

    conv2_1 = conv_wrapper(pooling1, 'conv2_1',is_training)
    conv2_2 = conv_wrapper(conv2_1, 'conv2_2',is_training)
    conv2_3 = conv_wrapper(conv2_2, 'conv2_3',is_training)
    pooling2 = pooling_wrapper(conv2_3, 'pool2')

    conv3_1 = conv_wrapper(pooling2, 'conv3_1',is_training)
    conv3_2 = conv_wrapper(conv3_1, 'conv3_2',is_training)
    conv3_3 = conv_wrapper(conv3_2, 'conv3_3',is_training)
    pooling3 = pooling_wrapper(conv3_3, 'pool3')

    # conv1_1 = tf.layers.conv2d(inputs,
    #                             32, #output channel number
    #                             (3,3),#kernel size
    #                             padding = 'same',
    #                             activation = activation,
    #                             kernel_initializer = kernel_initializer,
    #                             trainable=False,  #frozen, not training
    #                             name = 'conv1_1')
    # conv1_2 = tf.layers.conv2d(conv1_1,
    #                             32, #output channel number
    #                             (3,3),#kernel size
    #                             padding = 'same',
    #                             activation = activation,
    #                             kernel_initializer=kernel_initializer,
    #                             trainable=False,
    #                             name = 'conv1_2')
    #
    # conv1_3 = tf.layers.conv2d(conv1_2,
    #                            32,  # output channel number
    #                            (3, 3),  # kernel size
    #                            padding='same',
    #                            activation=activation,
    #                            kernel_initializer=kernel_initializer,
    #                            trainable=False,
    #                            name='conv1_3')

    # output - 16x16
    # pooling1 = tf.layers.max_pooling2d(conv1_3,
    #                                    (2,2), # kernel size
    #                                    (2,2), # stride
    #                                    name = 'pool1' )



    # conv2_1 = tf.layers.conv2d(pooling1,
    #                             32, #output channel number
    #                             (3,3),#kernel size
    #                             padding = 'same',
    #                             activation = activation,
    #                             kernel_initializer=kernel_initializer,
    #                             trainable=False,
    #                             name = 'conv2_1')
    #
    #
    # conv2_2 = tf.layers.conv2d(conv2_1,
    #                             32, #output channel number
    #                             (3,3),#kernel size
    #                             padding = 'same',
    #                             activation = activation,
    #                             kernel_initializer=kernel_initializer,
    #                             trainable=False,
    #                             name = 'conv2_2')
    #
    # conv2_3 = tf.layers.conv2d(conv2_2,
    #                            32,  # output channel number
    #                            (3, 3),  # kernel size
    #                            padding='same',
    #                            activation=activation,
    #                            kernel_initializer=kernel_initializer,
    #                            trainable=False,
    #                            name='conv2_3')
    #
    # # output - 8x8
    # pooling2 = tf.layers.max_pooling2d(conv2_3,
    #                                    (2,2), # kernel size
    #                                    (2,2), # stride
    #                                    name = 'pool2' )
    #
    # conv3_1 = tf.layers.conv2d(pooling2,
    #                             32, #output channel number
    #                             (3,3),#kernel size
    #                             padding = 'same',
    #                             activation = activation,
    #                             kernel_initializer=kernel_initializer,
    #                             name = 'conv3_1')
    # conv3_2 = tf.layers.conv2d(conv3_1,
    #                             32, #output channel number
    #                             (3,3),#kernel size
    #                             padding = 'same',
    #                             activation = activation,
    #                             kernel_initializer=kernel_initializer,
    #                             name = 'conv3_2')
    # conv3_3 = tf.layers.conv2d(conv3_2,
    #                            32,  # output channel number
    #                            (3, 3),  # kernel size
    #                            padding='same',
    #                            activation=activation,
    #                            kernel_initializer=kernel_initializer,
    #                            name='conv3_3')
    #
    # # output - 4x4x32
    # pooling3 = tf.layers.max_pooling2d(conv3_3,
    #                                    (2,2), # kernel size
    #                                    (2,2), # stride
    #                                    name = 'pool3' )

    # [None, 4x4x32]
    flatten = tf.layers.flatten(pooling3)
    return flatten

#从convnet中返回flatten
#可以尝试使用不同的激活函数 sigmoid 比 relu 要慢，精度也低

# 未使用图像增强时的输入为x_image
#flatten = convnet(x_image, tf.nn.relu, None)
# 使用图像增强后为增强后的图像
flatten = convnet(data_aug_3,is_training, tf.nn.relu, None)

# flatten = convnet(x_image, tf.nn.sigmoid)
# 全连接层,对应到10个类
y_ =tf.layers.dense(flatten, 10)

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
    # train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    train_op = tf.train.GradientDescentOptimizer(1e-04).minimize(loss)

# for tensorboard display
loss_summary = tf.summary.scalar('loss' ,loss)
# 'loss': <10, 1.1> <20, 1.08>
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# from (-1,1) to (0,255)
# 不做图像增强时的语句
# source_image = (x_image + 1) * 127.5
# inputs_summary = tf.summary.image('inputs_image', source_image)

# 做图像增强，此时因为前面没有归一化，所以不需要 *127.5的操作
inputs_summary = tf.summary.image('inputs_image', x_image)


#merge all above summary
merged_summary = tf.summary.merge_all()
#merge specified summary
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

LOG_DIR = '.'
run_label = 'vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

# save model for fine tuning
model_dir = os.path.join(run_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# for model saving
saver = tf.train.Saver()

# restore model
model_name = 'ckp-4400'
model_path = os.path.join(model_dir, model_name)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100
#compute summary to be displayed on tensorboard every 100 steps
output_summary_every_steps = 100

#save model every 100 steps
output_model_every_steps = 100

with tf.Session() as sess:
    sess.run(init)
    # create writer
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    #get fixed test data for inference, so as to get test accuracy to be displayed on tensorboard
    fixed_test_batch_data, fixed_test_batch_label = test_data.next_batch(batch_size)

    # using restored model
    if os.path.exists(model_path + '.index'):
        saver.restore(sess, model_path)
        print ('model restored from %s' % model_path)
    else:
        print ('model %s does not exist' % model_path)

    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        # whether to calculate merge summary
        eval_ops = [loss, accuracy, train_op]
        should_output_summary = ((i+1) % output_summary_every_steps == 0)
        #如果不需要计算的时候， 是否需要从列表中去掉merged_summary?
        if(should_output_summary):
            eval_ops.append(merged_summary)

        eval_ops_result = sess.run(
                                    eval_ops,
                                    feed_dict={
                                        x:batch_data,
                                        y:batch_labels,
                                        is_training:True}
                                        )
        loss_val, accu_val = eval_ops_result[0:2]
        if should_output_summary:
            train_summary_str = eval_ops_result[-1]
            train_writer.add_summary(train_summary_str, i+1)
            #get test result for display
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict={
                                            x: fixed_test_batch_data,
                                            y: fixed_test_batch_label,
                                            is_training: False
                                        })[0]
            test_writer.add_summary(test_summary_str, i+1)

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
                                                   y: test_batch_lables,
                                                   is_training: False})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test] Step: %d, acc: %4.5f' % (i+1, test_acc))

        # saving model (只保存最近的5次模型， tensor flow saver的机制)
        if (i+1) % output_model_every_steps == 0:
            saver.save(sess, os.path.join(model_dir, 'ckp-%d' % (i+1)))
            print ('model saved to ckp-%05d' % (i+1))