import tensorflow as tf
import os

from datetime import datetime

from sn_input_numpy import *
from sn_visualization import *
from output_to_png import *


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def activation(x):
    return lrelu(x)


def sigm(x):
    return tf.sigmoid(x)


def relu(x):
    return tf.nn.relu(x)


def lrelu(x):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= 0.1 * negative_part
    return x


def learn_kernels(kernel_name, batch_part='all', iterations=5000):
    my_input = NumpyInput('info2.txt')

    in_vectors = tf.placeholder(tf.float32, shape=[None, 64 * 64])

    conv1_kernel_size = 12
    conv2_kernel_size = 8

    conv1_kernel_num = 16 # 16
    conv2_kernel_num = 1

    sz1 = 4 # 4
    sz2 = 4

    w_conv1_1 = tf.Variable(tf.constant(0.0, shape=[conv1_kernel_size, conv1_kernel_size, 1, 8]), name='conv1_w_sharp')
    w_conv1_2 = tf.Variable(tf.constant(0.0, shape=[conv1_kernel_size, conv1_kernel_size, 1, 8]), name='conv1_w_blurry')
    w_conv1 = tf.concat(3, [w_conv1_1, w_conv1_2], name='w_conv_1')
    #w_conv1 = w_conv1_1
    #w_conv1 = weight_variable([conv1_kernel_size, conv1_kernel_size, 1, conv1_kernel_num], 'w_conv_1')
    b_conv1 = bias_variable([conv1_kernel_num], 'b_conv_1')
    b_unconv1 = bias_variable([1], 'b_unconv_1')

    w_conv2 = weight_variable([conv2_kernel_size, conv2_kernel_size, conv1_kernel_num, conv2_kernel_num], 'w_conv_2')
    b_conv2 = bias_variable([conv2_kernel_num], 'b_conv_2')
    b_unconv2 = bias_variable([conv1_kernel_num], 'b_unconv_2')

    in_images = tf.reshape(in_vectors, [-1, 64, 64, 1])

    h_conv1 = activation(tf.nn.conv2d(in_images,
                                      w_conv1,
                                      strides=[1, 2, 2, 1],
                                      padding='SAME') + b_conv1)
    h_conv2 = activation(tf.nn.conv2d(h_conv1,
                                      w_conv2,
                                      strides=[1, 4, 4, 1],
                                      padding='SAME') + b_conv2)

    h_unconv2 = activation(tf.nn.conv2d_transpose(h_conv2,
                                                  w_conv2,
                                                  output_shape=[25, 32, 32, conv1_kernel_num],
                                                  strides=[1, 4, 4, 1],
                                                  padding='SAME') + b_unconv2)
    result = activation(tf.nn.conv2d_transpose(h_unconv2,
                                               #h_conv1,
                                               w_conv1,
                                               output_shape=[25, 64, 64, 1],
                                               strides=[1, 2, 2, 1],
                                               padding='SAME') + b_unconv1)

    kernels1_on_grid = put_kernels_on_grid(w_conv1, sz1, sz2)
    kernels2_stack = tf.reshape(w_conv2, [conv2_kernel_size, conv2_kernel_size, 1, -1])
    kernels2_on_grid = put_kernels_on_grid(kernels2_stack, conv2_kernel_num, conv1_kernel_num)
    results_on_grid = put_kernels_on_grid(tf.transpose(result, (1, 2, 3, 0)), 5, 5)

    penalty = tf.mul(tf.nn.l2_loss(w_conv1), 2e-4)
    loss = tf.reduce_mean(tf.square(result - in_images))
    total_loss = loss# + penalty

    tf.summary.scalar('loss', loss)
    combined_summary = tf.summary.merge_all()

    train_speed = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(train_speed).minimize(total_loss)

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('./logdir', graph=session.graph)

    saver_sharp = tf.train.Saver({'conv1_w_sharp': w_conv1_1})
    saver_blurry = tf.train.Saver({'conv1_w_blurry': w_conv1_2})
    saver_sharp.restore(session, './checkpoints/conv1_w_sharp.ckpt')
    saver_blurry.restore(session, './checkpoints/conv1_w_blurry.ckpt')

    saver = tf.train.Saver(var_list={kernel_name: w_conv2})

    output_dir = './output_' + str(datetime.now()).replace(':', '-')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(iterations):
        if i % 100 == 0:
            test_labels, test_vectors = my_input.get_next_test_batch(batch_files_num=1, flag=batch_part)

            [kernels1, kernels2, results, summary, test_loss, test_penalty] = session.run([kernels1_on_grid, kernels2_on_grid, results_on_grid, combined_summary, loss, penalty],
                                                                               feed_dict={in_vectors: test_vectors})
            writer.add_summary(summary, i)

            print('iteration %d, test_loss=%f, test_penalty=%f' % (i, test_loss, test_penalty))

            write_bw_to_png(results, output_dir + '/results%06d.png' % i, 1)
            write_heatmap_to_png(kernels1, output_dir + '/kernels_1_%06d.png' % i, 5)
            write_heatmap_to_png(kernels2, output_dir + '/kernels_2_%06d.png' % i, 2)


        train_labels, train_vectors = my_input.get_next_train_batch(batch_files_num=1, flag=batch_part)

        [step] = session.run([train_step], feed_dict={in_vectors: train_vectors, train_speed: 1e-3})

    saver.save(session, './checkpoints/' + kernel_name + '.ckpt')

    session.close()

def main():
    learn_kernels('conv2_w_sharp', batch_part='first')
    #learn_kernels('conv2_w_blurry', batch_part='second')

if __name__ == '__main__':
    main()
