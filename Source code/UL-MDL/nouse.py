import numpy as np
#,def,gen_A(num_classes,t,adj_file):
import pickle
import matplotlib.pyplot as plt

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles
real_cov = np.array([[.8, .3],
                   [.3, .4]])
rng = np.random.RandomState(0)
X = rng.multivariate_normal(mean=[0, 0],
                             cov=real_cov,
                             size=500)



print(X)
print(X.shape)
cov = EmpiricalCovariance().fit(X)
print(cov.covariance_.shape)
