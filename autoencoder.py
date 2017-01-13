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

    kernel_num = 8
    sz1 = 2
    sz2 = 4

    w_conv1 = weight_variable([12, 12, 1, kernel_num], 'w_conv_1')
    b_conv1 = bias_variable([kernel_num], 'b_conv_1')
    b_unconv1 = bias_variable([1], 'b_unconv_1')

    in_images = tf.reshape(in_vectors, [-1, 64, 64, 1])

    h_conv1 = activation(tf.nn.conv2d(in_images,
                                      w_conv1,
                                      strides=[1, 6, 6, 1],
                                      padding='SAME') + b_conv1)

    result = activation(tf.nn.conv2d_transpose(h_conv1,
                                               w_conv1,
                                               output_shape=[25, 64, 64, 1],
                                               strides=[1, 6, 6, 1],
                                               padding='SAME') + b_unconv1)

    kernels_on_grid = put_kernels_on_grid(w_conv1, sz1, sz2)
    results_on_grid = put_kernels_on_grid(tf.transpose(result, (1, 2, 3, 0)), 5, 5)

    penalty = tf.mul(tf.nn.l2_loss(w_conv1), 2e-4)
    loss = tf.reduce_mean(tf.square(result - in_images))
    total_loss = loss + penalty

    tf.summary.scalar('loss', loss)
    tf.summary.image('original', in_images, max_outputs=2)
    tf.summary.image('kernels', kernels_on_grid)
    tf.summary.image('results', result, max_outputs=2)
    combined_summary = tf.summary.merge_all()

    train_speed = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(train_speed).minimize(total_loss)

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('./logdir', graph=session.graph)
    saver = tf.train.Saver(var_list={kernel_name: w_conv1})

    output_dir = './output_' + str(datetime.now()).replace(':', '-')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(iterations):
        if i % 100 == 0:
            test_labels, test_vectors = my_input.get_next_test_batch(batch_part)

            [kernels, results, summary, test_loss, test_penalty] = session.run([kernels_on_grid, results_on_grid, combined_summary, loss, penalty],
                                                                               feed_dict={in_vectors: test_vectors})
            writer.add_summary(summary, i)

            print('iteration %d, test_loss=%f, test_penalty=%f' % (i, test_loss, test_penalty))

            write_bw_to_png(results, output_dir + '/results%06d.png' % i, 1)
            write_heatmap_to_png(kernels, output_dir + '/kernels%06d.png' % i, 5)

        train_labels, train_vectors = my_input.get_next_train_batch(batch_part)

        [step] = session.run([train_step], feed_dict={in_vectors: train_vectors, train_speed: 1e-3})

    saver.save(session, './checkpoints/' + kernel_name + '.ckpt')

    session.close()

def main():
    #learn_kernels('conv1_w_sharp', batch_part='first')
    learn_kernels('conv1_w_blurry', batch_part='second')

if __name__ == '__main__':
    main()
