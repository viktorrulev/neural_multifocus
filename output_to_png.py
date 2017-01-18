import png
import numpy as np


def write_heatmap_to_png(a, filename, scale=1):
    a_shape = np.shape(a)
    colored = np.ndarray([a_shape[1] * scale, a_shape[2] * scale, 3], np.uint8)

    a_max = np.max(a)
    a_min = np.min(a)

    abs_max = max(abs(a_max), abs(a_min))

    for i in range(a_shape[1]):
        for j in range(a_shape[2]):
            val = a[0, i, j, 0]

            for n in range(scale):
                for m in range(scale):
                    ind_i = i * scale + n
                    ind_j = j * scale + m
                    colored[ind_i, ind_j, 0] = 0
                    colored[ind_i, ind_j, 1] = 0
                    colored[ind_i, ind_j, 2] = 0

                    if val > 0:
                        colored[ind_i, ind_j, 0] = abs(val * (255 / (abs_max + 0.001)))
                    else:
                        colored[ind_i, ind_j, 2] = abs(val * (255 / (abs_max + 0.001)))

    png.from_array(colored, 'RGB').save(filename)


def write_bw_to_png(a, filename, scale=1):
    a_shape = np.shape(a)
    bw = np.ndarray([a_shape[1] * scale, a_shape[2] * scale], np.uint8)

    a_min = np.min(a)
    a_max = np.max(a)

    for i in range(a_shape[1]):
        for j in range(a_shape[2]):
            val = a[0, i, j, 0]

            for n in range(scale):
                for m in range(scale):
                    ind_i = i * scale + n
                    ind_j = j * scale + m
                    bw[ind_i, ind_j] = (val - a_min) * (255 / (a_max - a_min))

    png.from_array(bw, 'L').save(filename)