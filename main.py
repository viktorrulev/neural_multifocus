import tensorflow as tf

from sn_input_numpy import *
from sn_visualization import *
from output_to_png import *


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def lrelu(x):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= 0.1 * negative_part
    return x


def main():
    my_input = NumpyInput('info2.txt')

    in_labels = tf.placeholder(tf.float32, shape=[None, 2], name='true_labels')
    in_vectors = tf.placeholder(tf.float32, shape=[None, 64 * 64])

    conv1_kernel_size = 12
    conv1_kernel_num = 16
    conv2_kernel_size = 12
    conv2_kernel_num = 16

    fc1_num = 500
    fc2_num = 300

    in_images = tf.reshape(in_vectors, [-1, 64, 64, 1])

    w_conv1_1 = tf.Variable(tf.constant(0.0, shape=[conv1_kernel_size, conv1_kernel_size, 1, 8]), name='conv1_w_sharp')
    w_conv1_2 = tf.Variable(tf.constant(0.0, shape=[conv1_kernel_size, conv1_kernel_size, 1, 8]), name='conv1_w_blurry')
    w_conv1 = tf.concat(3, [w_conv1_1, w_conv1_2], name='w_conv_1')

    b_conv1 = bias_variable([conv1_kernel_num], 'b_conv_1')
    w_conv2 = weight_variable([conv2_kernel_size, conv2_kernel_size, conv1_kernel_num, conv2_kernel_num], 'w_conv_2')
    b_conv2 = weight_variable([conv2_kernel_num], 'b_conv_2')

    h_conv1 = lrelu(conv2d(in_images, w_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)
    h_conv2 = lrelu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = h_conv2

    result_side = int(h_pool2.get_shape()[1])

    w_fc1 = weight_variable([result_side * result_side * conv2_kernel_num, fc1_num], 'w_full_1')
    b_fc1 = bias_variable([fc1_num], 'b_full_1')
    w_fc2 = weight_variable([fc1_num, fc2_num], 'w_full_2')
    b_fc2 = bias_variable([fc2_num], 'b_full_2')
    w_fc3 = weight_variable([fc2_num, 2], 'w_full_3')
    b_fc3 = bias_variable([2], 'b_full_3')

    h_pool2_flat = tf.reshape(h_pool2, [-1, result_side * result_side * conv2_kernel_num], name='flat')

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
    labels = tf.add(tf.matmul(h_fc2, w_fc3), b_fc3, name='nn_labels')

    with tf.name_scope('penalty'):
        w_conv1_loss = tf.nn.l2_loss(w_conv1)
        b_conv1_loss = tf.nn.l2_loss(b_conv1)
        w_conv2_loss = tf.nn.l2_loss(w_conv2)
        b_conv2_loss = tf.nn.l2_loss(b_conv2)
        w_fc1_loss = tf.nn.l2_loss(w_fc1)
        b_fc1_loss = tf.nn.l2_loss(b_fc1)
        w_fc2_loss = tf.nn.l2_loss(w_fc2)
        b_fc2_loss = tf.nn.l2_loss(b_fc2)
        w_fc3_loss = tf.nn.l2_loss(w_fc3)
        b_fc3_loss = tf.nn.l2_loss(b_fc3)

        reg_penalty = w_conv1_loss + b_conv1_loss + w_conv2_loss + b_conv2_loss\
            + w_fc1_loss + b_fc1_loss + w_fc2_loss + b_fc2_loss + w_fc3_loss + b_fc3_loss

        reg_penalty_lambda = 1e-5

        total_reg_penalty = tf.mul(reg_penalty, reg_penalty_lambda)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, in_labels), name='entropy')
    total_loss = cross_entropy + total_reg_penalty

    train_speed = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(train_speed).minimize(total_loss)

    correct_prediction = tf.equal(tf.argmax(in_labels, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.name_scope('summary'):
        kernels_on_grid = put_kernels_on_grid(w_conv1, 4, 4)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('reg_penalty', reg_penalty)
        #tf.summary.image('features', kernels_on_grid)
        #tf.summary.image('batch', in_images)
        combined_summary = tf.summary.merge_all()

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('./logdir', graph=session.graph)
    saver_sharp = tf.train.Saver({'conv1_w_sharp': w_conv1_1})
    saver_blurry = tf.train.Saver({'conv1_w_blurry': w_conv1_2})

    saver_sharp.restore(session, './checkpoints/conv1_w_sharp.ckpt')
    saver_blurry.restore(session, './checkpoints/conv1_w_blurry.ckpt')

    spd = 2e-4
    spd_decrease = 0.5
    i = 0

    for epoch in range(10):
        if epoch != 0:
            spd = spd * spd_decrease

        print('epoch #%d' % (epoch + 1))
        print('train speed: %f' % spd)

        for batch in range(my_input.train_batch_num):
            if i % 100 == 0:
                test_labels, test_vectors = my_input.get_next_test_batch()
                [acc, summary, ce, pen] = session.run([accuracy, combined_summary, cross_entropy, total_reg_penalty],
                                                      feed_dict={in_labels: test_labels, in_vectors: test_vectors})

                writer.add_summary(summary, i)
                print('iteration %05d, accuracy = %f, cross-entropy = %f, penalty = %f' % (i, acc, ce, pen))

            train_labels, train_vectors = my_input.get_next_train_batch()
            step = session.run([train_step],
                               feed_dict={in_labels: train_labels, in_vectors: train_vectors, train_speed: spd})

            i = i + 1


if __name__ == '__main__':
    main()
