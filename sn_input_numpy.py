import numpy as np
from random import shuffle
import math as math


def preprocess_scale(fragment):
    return np.clip(fragment, 0.0, 256.0) / 256.0


def preprocess_std(fragment):
    clipped_fragment = np.clip(fragment, 0.0, 256.0)
    return (clipped_fragment - np.mean(clipped_fragment)) / np.std(clipped_fragment)


def preprocess_log(fragment):
    return np.log(1.0 + np.clip(fragment, 0.0, 255.0)) / np.log(256.0)


def preprocess_fragment(fragment):
    return preprocess_scale(fragment)


class NumpyInput:
    BATCH_SIZE = 50
    LABEL_SIZE = 2
    IMAGE_SIZE = 64 * 64

    TRAIN_BATCH_FILES = 1
    TEST_BATCH_FILES = 2 #8

    TRAIN_FIXED = False
    TEST_FIXED = True

    def __init__(self, info_filename):
        f = open(info_filename, 'r')

        file_lines = f.read().splitlines()

        if '' in file_lines:
            file_lines.remove('')

        batch_num = int(file_lines[0])
        file_list = file_lines[1:]

        shuffle(file_list)

        test_percentage = 0.05
        self.test_batch_num = math.ceil(test_percentage * batch_num)
        self.train_batch_num = batch_num - self.test_batch_num

        self.test_files = file_list[0:self.test_batch_num]
        self.train_files = file_list[self.test_batch_num:batch_num]

        self.next_test_batch = 0
        self.next_train_batch = 0

        f.close()

        print('%d train batches, %d test batches' % (self.train_batch_num, self.test_batch_num))

    def get_next_train_batch(self, batch_files_num=1, flag='all'):
        actual_batch_size = 0
        if flag == 'first' or flag == 'second':
            actual_batch_size = self.BATCH_SIZE // 2
        else:
            actual_batch_size = self.BATCH_SIZE

        labels = np.ndarray((batch_files_num * actual_batch_size, self.LABEL_SIZE), np.float32)
        images = np.ndarray((batch_files_num * actual_batch_size, self.IMAGE_SIZE), np.float32)

        if self.TRAIN_FIXED:
            self.next_train_batch = 0

        batch_first = 0
        batch_last = 0

        if flag == 'all':
            batch_first = 0
            batch_last = self.BATCH_SIZE
        elif flag == 'first':
            batch_first = 0
            batch_last = self.BATCH_SIZE // 2
        elif flag == 'second':
            batch_first = self.BATCH_SIZE // 2
            batch_last = self.BATCH_SIZE
        else:
            print('incorrect flag: %s' % flag)
            exit(0)


        for times in range(batch_files_num):
            f = open(self.train_files[self.next_train_batch], 'rb')

            data = np.fromfile(f, dtype=np.uint8)
            data_2d = np.reshape(data, [-1, self.LABEL_SIZE + self.IMAGE_SIZE])
            ind1_first = times * actual_batch_size
            ind1_last = (times + 1) * actual_batch_size
            labels[ind1_first:ind1_last, :] = data_2d[batch_first:batch_last, 0:self.LABEL_SIZE].astype(float)
            fragment = data_2d[batch_first:batch_last, self.LABEL_SIZE:(self.LABEL_SIZE + self.IMAGE_SIZE)].astype(float)
            images[ind1_first:ind1_last, :] = preprocess_fragment(fragment)

            self.next_train_batch = (self.next_train_batch + 1) % self.train_batch_num

            f.close()

        return labels, images

    def get_next_test_batch(self, batch_files_num=1, flag='all'):
        actual_batch_size = 0
        if flag == 'first' or flag == 'second':
            actual_batch_size = self.BATCH_SIZE // 2
        else:
            actual_batch_size = self.BATCH_SIZE

        labels = np.ndarray((batch_files_num * actual_batch_size, self.LABEL_SIZE), np.float32)
        images = np.ndarray((batch_files_num * actual_batch_size, self.IMAGE_SIZE), np.float32)

        if self.TEST_FIXED:
            self.next_test_batch = 0

        batch_first = 0
        batch_last = 0

        if flag == 'all':
            batch_first = 0
            batch_last = self.BATCH_SIZE
        elif flag == 'first':
            batch_first = 0
            batch_last = self.BATCH_SIZE // 2
        elif flag == 'second':
            batch_first = self.BATCH_SIZE // 2
            batch_last = self.BATCH_SIZE
        else:
            print('incorrect flag: %s' % flag)
            exit(0)

        for times in range(batch_files_num):
            f = open(self.test_files[self.next_test_batch], 'rb')

            data = np.fromfile(f, dtype=np.uint8)
            data_2d = np.reshape(data, [-1, self.LABEL_SIZE + self.IMAGE_SIZE])
            ind1_first = times * actual_batch_size
            ind1_last = (times + 1) * actual_batch_size
            labels[ind1_first:ind1_last, :] = data_2d[batch_first:batch_last, 0:self.LABEL_SIZE].astype(float)
            fragment = data_2d[batch_first:batch_last, self.LABEL_SIZE:(self.LABEL_SIZE + self.IMAGE_SIZE)].astype(float)
            images[ind1_first:ind1_last, :] = preprocess_fragment(fragment)

            self.next_test_batch = (self.next_test_batch + 1) % self.test_batch_num

            f.close()

        return labels, images
