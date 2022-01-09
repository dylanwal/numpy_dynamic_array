import numpy as np
from numpy_da import DynamicArray

data = DynamicArray(shape=(5, 2), index_expansion=True)
for i in range(11):
    data[i, 0] = i
answer = np.column_stack((np.zeros(11), np.linspace(0, 10, 11)))

print("hi")
