# Numpy Dynamic Array

---
---
![PyPI](https://img.shields.io/pypi/v/numpy_dynamic_array)
![tests](https://raw.githubusercontent.com/dylanwal/numpy_dynamic_array/master/tests/badges/tests-badge.svg)
![coverage](https://raw.githubusercontent.com/dylanwal/numpy_dynamic_array/master/tests/badges/coverage-badge.svg)
![flake8](https://raw.githubusercontent.com/dylanwal/numpy_dynamic_array/master/tests/badges/flake8-badge.svg)
![downloads](https://img.shields.io/pypi/dm/numpy_dynamic_array)
![license](https://img.shields.io/github/license/dylanwal/numpy_dynamic_array)

Dynamically resizing Numpy array.

A dynamic array expands as you add more elements. So you don't need to determine the size ahead of time. The version
present here is focused on being compatible with the typical Numpy indexing and functions.

---

## Installation

For python >=3.9
```
pip install numpy_dynamic_array
```


For python 3.7-3.9, directly install from the python37 branch with
```
pip install https://github.com/dylanwal/numpy_dynamic_array/archive/python37.zip
```

---
---

## Usage

### Basics

```python
import numpy as np
from numpy_da import DynamicArray

data = DynamicArray(shape=2)
a = np.linspace(0, 9, 10)
data.append(a)  # requires resize (done automatically)

print(data)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

```

---

### Indexing

Setting values with indexing

```python
data = DynamicArray(shape=2)
data.append(np.linspace(0, 9, 10))
data[3] = 40
print(data)  # [0, 1, 2, 3, 40, 5, 6, 7, 8, 9]
```

Setting values with indexing (outside current array size)
Set `index_expansion=True`

```python
data = DynamicArray(shape=2, index_expansion=True)
data[5] = 1
print(data)  # [0, 0, 0, 0, 0, 1]
```

Setting values with slices (outside current array size)
Set `index_expansion=True`

```python
data = DynamicArray(shape=2, index_expansion=True)
data[5:8] = [1, 2, 3]
print(data)  # [0, 0, 0, 0, 0, 1, 2, 3]
```

---

### Operators

Equality:

```python
np_array = np.linspace(0, 9, 10)
data = DynamicArray()
data.append(np_array)
print(all(data == np_array))  # True
```

Addition:

```python
np_array = np.linspace(0, 9, 10)
data = DynamicArray()
data.append(np_array)
print(all((data + 7.5) == (np_array + 7.5)))  # True
```

Other supported oparators:
* floordiv (//)
* mod (%)
* mul (*)
* neg (-)
* pow (**)
* truediv(/)
* sub (-)
* len (len())


---
### Numpy Functions

```python
np_array = np.linspace(0, 9, 10)
data = DynamicArray()
data.append(np_array)
print(data.max())  # 9
```

```python
np_array = np.linspace(0, 9, 10)
data = DynamicArray()
data.append(np_array)
print(np.abs(data - 5))  # [5. 4. 3. 2. 1. 0. 1. 2. 3. 4.]
```


---
### Multidimensional arrays

Use the shape to specify initial ndarray with correct dimensions.

```python
data = DynamicArray(shape=(2, 2))  # requires resize
a = np.linspace(0, 9, 10).reshape(5, 2)
data.append(a)
```


---
<b> For more examples look at the tests folder. </b>
