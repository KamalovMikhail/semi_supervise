__author__ = 'Mikhail Kamalov'
import numpy as np

from graph_base_semi_supervise import BasedSemiSupervise
from scipy.sparse import csr_matrix

ssl = BasedSemiSupervise(method="d-max")

X = csr_matrix([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])


y = np.array([[0], [1], [2], [1]])

ssl.fit(X, y)

print("weights")
print(ssl.label_distributions_)
print("Answer")
print(ssl.predict(X))
print("X_")
print(ssl.X_)


