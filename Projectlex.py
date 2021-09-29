import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

rng = np.random.RandomState(1)
print(rng.rand(3, 2))
print('\n')
X = np.dot(rng.rand(2, 2), rng.randn(2, 5)).T
print(X)
