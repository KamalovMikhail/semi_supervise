__author__ = 'Mikhail Kamalov'
import numpy as np

from graph_base_semi_supervise import BasedSemiSupervise
from scipy.sparse import csr_matrix

ssl = BasedSemiSupervise()

X = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)


y = np.array([[0], [1], [2]])

ssl.fit(X, y)
print(ssl.label_distributions_)
print(ssl.predict(X))


