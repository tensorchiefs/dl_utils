import dl_utils as dl
import numpy as np

np.random.seed(42)
X = np.random.normal(0,1,(5,10))
print(dl.get_quantiles(X, 3))