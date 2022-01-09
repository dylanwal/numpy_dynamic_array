import numpy as np
from numpy_da import DynamicArray


data = DynamicArray(shape=5)

# index beyond array size (it will grow as needed)
for i in range(11):
    data[i] = i

print(data)
print(data == np.linspace(0, 10, 11))

# modify data with indexing
data[0] = -5
data[1] = data[1] * -1
data[-1] = 100  # index 15
data[2] += 1
print(data)

# modify slices of data
data[3:6] = [0, 0, 0]
print(data)

# operators
# element
print(data[0] == 1)
print(data[0] >= 1)

# slice
print(data[3:6] == [0, 0, 0])
print(data[3:6] + 1)

# whole array
print(data + 1)
print(data - 1)
print(data * -1)
print(data / 2)
print(data ** 2)
print(data % 2)

# numpy functions
print(data.max())
print(data.T)

# using 'append'
data.append(1)
data.append(np.linspace(1, 9, 10))
print(data)
