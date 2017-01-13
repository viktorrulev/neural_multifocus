import numpy as np


def write_for_matlab(a, filename):
    f = open(filename, 'w')
    a_shape = np.shape(a)

    f.write('m=[\n')
    for i in range(a_shape[1]):
        for j in range(a_shape[2]):
            f.write(str(np.asscalar(a[0, i, j, 0])))
            f.write(', ')

        f.write(';\n')
    f.write('];')

    f.close()