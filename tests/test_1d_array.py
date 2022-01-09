import pytest

import numpy as np
from numpy_da import DynamicArray


def test_initialization():
    data = DynamicArray()
    assert isinstance(data, DynamicArray)


# test append
def test_append_nparray():
    data = DynamicArray(shape=15)
    a = np.linspace(0, 9, 10)
    data.append(a)
    for i in range(10):
        assert a[i] == data[i]
    assert data.size == 10
    assert data.capacity == 5


def test_append_nparray_grow():
    data = DynamicArray(shape=2)  # requires resize
    a = np.linspace(0, 9, 10)
    data.append(a)
    assert data.size == 10
    assert all(data == a)


def test_append_int():
    data = DynamicArray(shape=10)
    data.append(1)
    assert data.size == 1
    assert data == [1]
    assert data.capacity == 9


def test_append_int_grow():
    data = DynamicArray(shape=10)
    for i in range(15):
        data.append(1)  # requires resize
    assert data.size == 15
    assert all(data == np.ones(15))


def test_append_list():
    data = DynamicArray(shape=10)
    a = [0, 1, 2, 3, 4, 5, 6]
    data.append(a)
    assert data.size == 7
    assert all(data == a)
    assert data.capacity == 3


def test_append_list_grow():
    data = DynamicArray(shape=5)
    a = [0, 1, 2, 3, 4, 5, 6]
    data.append(a)  # requires resize
    assert data.size == 7
    assert all(data == np.linspace(0, 6, 7))


@pytest.fixture(scope="session")
def np_array():
    return np.linspace(0, 9, 10)


# test operators
def test_equality(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all(data == np_array)


def test_add(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data + 7.5) == (np_array + 7.5))


def test_floordiv(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data // 3) == (np_array // 3))


def test_mod(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data % 3) == (np_array % 3))


def test_mul(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data * 3) == (np_array * 3))


def test_neg(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((-data) == (-np_array))


def test_pow(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data ** 3) == (np_array ** 3))


def test_truediv(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data / 3) == (np_array / 3))


def test_sub(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all((data - 3) == (np_array - 3))


def test_len(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert len(data) == len(np_array)


# index beyond array size (it will grow as needed)
def test_growing_array_by_index():
    data = DynamicArray(shape=2, index_expansion=True)
    for i in range(11):
        data[i] = i
    assert all(data == np.linspace(0, 10, 11))


def test_growing_array_by_slice():
    data = DynamicArray(shape=2, index_expansion=True)
    data[5:8] = [1, 2, 3]
    assert all(data == [0, 0, 0, 0, 0, 1, 2, 3])


def test_error_index_zero():
    data = DynamicArray(shape=2, index_expansion=True)
    data[5] = 1
    assert all(data == [0, 0, 0, 0, 0, 1])


# modify data with indexing
def test_indexing():
    data = DynamicArray()
    data.append(np.linspace(0, 9, 10))
    data[3] = 40
    assert data[3] == 40


def test_indexing_neg():
    data = DynamicArray()
    data.append(np.linspace(0, 9, 10))
    data[-1] = 100
    assert data[9] == 100


def test_indexing_plus_equal():
    data = DynamicArray()
    data.append(np.linspace(0, 9, 10))
    data[2] += 1
    assert data[2] == 3


def test_indexing_slice():
    data = DynamicArray()
    data.append(np.linspace(0, 9, 10))
    data[3:6] = [0, 0, 0]
    assert all(data == [0, 1, 2, 0, 0, 0, 6, 7, 8, 9])


def test_indexing_slice2():
    data = DynamicArray()
    data.append(np.linspace(0, 9, 10))
    data[3:6] = [0, 0, 0]
    assert all(data[3:6] == [0, 0, 0])


# Errors
def test_error_index_get():
    with pytest.raises(IndexError):
        data = DynamicArray()
        data.append(np.linspace(0, 9, 10))
        print(data[100])


def test_error_index_set():
    with pytest.raises(IndexError):
        data = DynamicArray()
        data.append(np.linspace(0, 9, 10))
        data[100] = 100


def test_error_append_dict():
    with pytest.raises(ValueError):
        data = DynamicArray()
        data.append({"fish": 1})


# numpy functions
def test_numpy_max(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert data.max() == np_array.max()


def test_numpy_item(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert data.item(2) == np_array.item(2)


def test_numpy_abs(np_array):
    data = DynamicArray()
    data.append(np_array)
    assert all(np.abs(data - 5) == np.abs(np_array - 5))
