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
import numpy as np

from scipy import sparse
from scipy.linalg import fractional_matrix_power as frac
from scipy.sparse import csr_matrix as sp_mat
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
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, 'X_')
        X_2d = check_array(X, accept_sparse = ['csc', 'csr', 'coo', 'dok',
                                               'bsr', 'lil', 'dia'])
        samples_for_predict = np.array(np.all((self.X_[:,None,:]==X_2d[None,:,:]),axis=-1).nonzero()).T.tolist()
        return self.label_distributions_[samples_for_predict[:,0]]

    def _get_method(self, X, y):
        if self.method == "pi":
            self.label_distributions_ = self.power_iteration(X, y, self.initial_vector_)
        else:
            raise ValueError("%s is not a valid method. Only pi"
                             " are supported at this time" % self.method)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        check_classification_targets(y)
        classes = np.nonzero(y)

        n_samples, n_classes = len(y), len(classes)

        # create diagonal matrix of degree of nodes
        D = sparse.dia_matrix(np.array(sp_mat.sum(self.X_, axis=0))[0])
        D_left = frac(D, - self.sigma)
        D_right = frac(D, 1 - self.sigma)

        B_ = (D_left.dot(self.X_)).dot(D_right)

        # create labeled data Z
        dimension = (n_samples, n_classes)
        labels = np.nonzero(y)
        ans_y = np.zeros(dimension)

        for l in labels:
            ans_y[l][y[l] - 1] = 1

        Z_ = (self.sigma / (1 + self.sigma)) * ans_y
        self.initial_vector_ = np.ones(dimension) / n_classes
        self._get_method(B_, Z_)
        return self

    def power_iteration(self, B, Z, x):
        iteration = 0
        y = np.copy(x)
        while iteration < self.max_iter:
            y = ((1 / (1 + self.mu)) * (B.T.dot(x))) + Z
            x = np.copy(y)
            iteration += 1
        return x
