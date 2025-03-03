import numpy as np


def print_corr(t):
    il = str(t.tolist()).replace('[', '{').replace(']', '}')
    print('Tensor corr {};'.format(il))


scalar_0 = np.array(1.23)
scalar_1 = np.array(2.34)

vector_0 = np.array([1.23, 2.34, 3.45])
vector_1 = np.array([4.56, 5.67, 6.78])

matrix_0 = np.array([[1.23, 2.34], [3.45, 4.56]])
matrix_1 = np.array([[5.67, 6.78], [7.89, 8.90]])

tensor3_0 = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]])
tensor3_1 = np.array([[[9.9, 10.10], [11.11, 12.12]],
                      [[13.13, 14.14], [15.15, 16.16]]])

tensor4_0 = np.array([[[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]],
                      [[[9.9, 10.10], [11.11, 12.12]],
                       [[13.13, 14.14], [15.15, 16.16]]]])
tensor4_1 = np.array([[[[17.17, 18.18], [19.19, 20.20]],
                       [[21.21, 22.22], [23.23, 24.24]]],
                      [[[25.25, 26.26], [27.27, 28.28]],
                       [[29.29, 30.30], [31.31, 32.32]]]])

print_corr(np.einsum('ij,jkl->ikl', matrix_0, tensor3_0))
