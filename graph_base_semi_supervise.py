__author__ = 'mikhail'


from scipy.sparse import csr_matrix as sp_mat
from scipy import sparse
import numpy as np
from scipy.linalg import fractional_matrix_power as frac
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets

class BasedSemiSupervise():

    def __init__(self, method='gs', sigma=0.5,  max_iter=100, mu=0.5):
       self.max_iter = max_iter
       self.method = method
       self.sigma = sigma
       self.mu = mu

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()


    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        check_classification_targets(y)
        classes = np.unique(y)
        classes = (classes[classes != 0])
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)


        # create diagonal matrix of degree of nodes
        D = sparse.dia_matrix(np.array(sp_mat.sum(self.X_, axis=0))[0])
        D_left = frac(D, - self.sigma)
        D_right = frac(D, 1 - self.sigma)

        B = (D_left.dot(self.X_)).dot(D_right)

        # create labeled data Z
        dimension = (n_samples, n_classes)
        labels = np.nonzero(y)
        ans_y = np.zeros(dimension)

        for l in labels:
            ans_y[l][y[l] - 1] = 1

        Z = (self.sigma / (1 + self.sigma)) * ans_y
        x = np.ones(dimension) / n_classes
        self.label_distributions_ = np.zeros(dimension)

        if self.method == 'pi':
            self.label_distributions_ = self.power_iteration(B, Z, x)





    def power_iteration(self, B, Z, x):
        iter = 0
        while iter < self.max_iter:
            print(iter)
            y = ((1 / (1 + self.mu)) * (B.T.dot(x))) + Z
            x = np.copy(y)
            iter += 1

        return x
