import numpy as np

two_dimensions = [[0] * 5 for i in range(3)]

print('Reshape to 5 list with 3 data')
print(np.array(two_dimensions).reshape(5, 3))

print('Reshape to 3 list with 5 data')
print(np.array(two_dimensions).reshape(3, 5))

