import dl_utils as dl
import numpy as np

np.random.seed(42)
X = np.random.normal(0,1,(5,10))
print(dl.get_quantiles(X, 3))


import dl_utils as dl
print(dl.joke())
print(dl.get_available_gpus())

import numpy as np
X = np.random.normal(0,1,(200,19))
res = dl.get_VI_MAPS(X)
print(res)
pred_label = res[3]

res2 = dl.get_VI_MAPS(X, label=pred_label)
print(res2)


res3 = dl.get_VI_MAPS(X, label=0)
print(res3)
