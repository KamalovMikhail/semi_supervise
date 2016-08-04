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

def d_iteration(B, np.ndarray Z, int type, int max_iter, float mu,):
        cdef unsigned int n_samples = Z.shape[0]
        cdef unsigned int n_classes = Z.shape[1]
        cdef np.ndarray H = np.zeros((n_samples, n_classes))
        cdef np.ndarray Fy = np.copy(Z)
        cdef np.ndarray Hy = np.copy(H)
        cdef np.ndarray e = np.zeros(n_samples)
        cdef np.ndarray probability = np.empty(n_samples)
        cdef unsigned int iteration, i, j = 0

        while iteration < max_iter:
            for s in range(0, n_samples):
                if type == 1:
                    j = np.argmax(np.sum(Z, axis=1))
                elif type == 0:
                    j = s

                e[j] = 1
                probability = B.dot(e)
                for i in np.nonzero(B[j])[0]:
                    Fy[i] = Z[i] + (1 / (1 + mu)) * Z[j] * probability[i]
                    Hy[i] = H[i] + Z[i]
                e *= 0
                Fy[j] *= 0
                Hy[j] = H[j] + Z[j]
                H = np.copy(Hy)
                Z = np.copy(Fy)
            iteration += 1
        return Hy