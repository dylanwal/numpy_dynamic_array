from numpy_da import DynamicArray


data = DynamicArray(shape=5)

for i in range(15):
    data[i] = i

print(data.data)

