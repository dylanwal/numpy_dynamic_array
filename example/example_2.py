import numpy as np
from numpy_da import DynamicArray


data = DynamicArray(shape=(8, 2))

# add data
data.add(np.linspace(0, 9, 10).reshape(5, 2))
data.add(np.linspace(0, 9, 10).reshape(5, 2))


# modify current data
data[1] = [55, 55]
data[3:8] = np.ones((5, 2)) * -1
data[-1] = [100, 100]

# index outside current size
if data.index_expansion:
    data[0, 30] = 5
else:
    try:
        data[0, 30] = 5
    except IndexError:
        print("IndexError raise, and expected.")

# view data
print(data.data[5])
print()
print(data.data[6:10])
print()
print(data.data)

