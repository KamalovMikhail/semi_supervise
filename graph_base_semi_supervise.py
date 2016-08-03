#!python
#cython: language_level=3, boundscheck=False
"""
Graph based semi-supervise method.

Model Features
--------------

Examples
--------

Notes
-----
References:
[1] Avrachenkov, K., Gonçalves, P., Mishenin, A., & Sokol, M. (2012, April).
Generalized optimization framework for graph-based semi-supervised learning.
In Proceedings of SIAM Conference on Data Mining (SDM 2012) (Vol. 9).
"""

# Authors: Mikhail Kamalov <mkamalovv@gmail.com>;
#          Konstantin Avrachenkov <konstantin.avratchenkov@inria.fr>;
#          ALexey Mishenin <alexey.mishenin@gmail.com>;
import gauss_seidel_cython
import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix, dia_matrix
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class BasedSemiSupervise:

    def __init__(self, method='gs', sigma=0.5,  max_iter=100, mu=0.5):
        self.max_iter = max_iter
        self.method = method
        self.sigma = sigma
        self.mu = mu

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1) + 1

    def predict_proba(self, X):
        check_is_fitted(self, 'X_')
        return self.label_distributions_

    def _get_method_(self, X, y):
        if self.method == "pi":
            self.label_distributions_ = self.power_iteration(X.T, y, self.initial_vector_)
        elif self.method == "gs":
            #implementation Gauss-Seidel for the dense and sparse representations completely written in cython
            self.label_distributions_ = gauss_seidel_cython.gauss_seidel(X.T, y, self.initial_vector_, self.max_iter, self.mu)
        else:
            raise ValueError("%s is not a valid method. Only pi"
                             " are supported at this time" % self.method)

    def fit(self, X, y):
        check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo', 'dok',
                        'bsr', 'lil', 'dia'])
        check_array(X, accept_sparse=['csc', 'csr', 'coo', 'dok',
                        'bsr', 'lil', 'dia'])
        self.X_ = X
        check_classification_targets(y)
        classes = np.nonzero(y)

        n_samples, n_classes = len(y), len(classes)
        # create diagonal matrix of degree of nodes
        if sparse.isspmatrix(self.X_):
            B_ = self.X_.copy()
            D = np.array(csr_matrix.sum(self.X_, axis=1)).T[0]
        else:
            B_ = np.copy(self.X_)
            D = np.array(np.sum(self.X_, axis=1))

        # if  (- self.sigma) and (self.sigma - 1) doesn't equals we have different diagonal matrix at the left and right sides
        if (- self.sigma) == (self.sigma - 1):
            D_left = D_right = np.power(D, - self.sigma)
        else:
            D_left = np.power(D, - self.sigma)
            D_right = np.power(self.sigma - 1)

        # M_ = D_left.dot(B_)
        for i, d in enumerate(D_left):
            B_[i, :] *= d
        # B_ = M_.dot(D_right)
        for i, d in enumerate(D_right):
            B_[:, i] *= d
        # create labeled data Z
        dimension = (n_samples, n_classes)
        labels = np.nonzero(y)
        ans_y = np.zeros(dimension)
        for l in labels[0]:
            ans_y[l][y[l] - 1] = 1

        Z_ = (self.sigma / (1 + self.sigma)) * ans_y
        self.initial_vector_ = np.ones(dimension) / n_classes
        self._get_method_(B_, Z_)
        return self

    def power_iteration(self, B, Z, x):
        iteration = 0
        y = np.copy(x)
        while iteration < self.max_iter:
            y = ((1 / (1 + self.mu)) * (B.dot(x))) + Z
            x = np.copy(y)
            iteration += 1
        return y

