import numpy as np
cimport numpy as np

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def gauss_seidel(B, np.ndarray Z, np.ndarray x, int max_iter, float mu, ):
        cdef unsigned int iteration, i, j = 0
        cdef unsigned int  n_classes = Z.shape[1]
        cdef np.ndarray sum_prev, sum_next = np.zeros((1, n_classes))
        cdef np.ndarray y = np.copy(x)
        cdef unsigned int n_samples = B.shape[0]

        while iteration < max_iter:
            for i in range(0, n_samples):
                sum_prev = np.zeros((1, n_classes))
                sum_next = np.zeros((1, n_classes))
                for j in range(0, n_samples):
                    if j < i:
                        sum_prev += np.multiply(y[j], B[i, j])
                    else:
                        sum_next += np.multiply(x[j], B[i, j])
                y[i] = (1 / (1 + mu)) * np.add(sum_prev, sum_next) + Z[i]
            iteration += 1
            x = np.copy(y)
        return y
