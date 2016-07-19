cimport numpy as np

def gauss_seidel(np.ndarray[double, ndim=2] B, np.ndarray[double, ndim=2] Z, np.ndarray[double, ndim=2] x,  np.ndarray[double, ndim=2] y, int max_iter, float mu):
        cdef unsigned int iteration , i, j = 0
        cdef float sum_prev, sum_next = 0
        cdef np.ndarray[long, ndim=2] current_samples
        P = B.T
        n_samples = B.shape[0]
        row, col = P.nonzero()
        print(iteration)
        print(max_iter)
        while iteration < max_iter:
            print(iteration)
            for i in range(0, n_samples):
                sum_prev = 0
                sum_next = 0
                current_samples = col[row == i]
                print(current_samples[0])
                for j in current_samples[0]:
                    if j < i:
                        sum_prev += y[j] * float(P[i, j])
                    else:
                        sum_next += x[j] * float(P[i, j])
                y[i] = (1 / (1 + mu)) * (sum_prev + sum_next) + Z[i]
                iteration += 1
                x = np.copy(y)
        return y