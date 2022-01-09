import pytest

import numpy as np
from numpy_da import DynamicArray


def test_initialization():
    data = DynamicArray()
    assert isinstance(data, DynamicArray)


# test append
def test_append_nparray():
    data = DynamicArray(shape=(15, 2))
    a = np.linspace(0, 9, 10).reshape(5, 2)
    data.append(a)
    for i in range(a.shape[0]):
        assert all(a[i] == data[i])
    assert data.size == 5
    assert data.capacity == 10


def test_append_nparray_grow():
    data = DynamicArray(shape=(2, 2))  # requires resize
    a = np.linspace(0, 9, 10).reshape(5, 2)
    data.append(a)
    assert data.size == 5
    assert np.array_equal(data, a)


def test_append_list():
    data = DynamicArray(shape=(10, 2), dtype="int64")
    a = [0, 1, 2, 3, 4, 5, 6]
    aa = [[i, i] for i in a]
    data.append(aa)
    assert data.size == 7
    assert np.array_equal(data, aa)
    assert data.capacity == 3


def test_append_list_grow():
    data = DynamicArray(shape=(5, 2), dtype="int64")
    a = [0, 1, 2, 3, 4, 5, 6]
    aa = [[i, i] for i in a]
    data.append(aa)  # requires resize
    assert data.size == 7
    assert np.array_equal(data, aa)


@pytest.fixture(scope="session")
def np_array():
    return np.linspace(0, 9, 10).reshape(5, 2)


# test operators
def test_equality(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data, np_array)


def test_add(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal((data + 7.5), (np_array + 7.5))


def test_floordiv(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data // 3, np_array // 3)


def test_mod(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data % 3, np_array % 3)


def test_mul(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data * 3, np_array * 3)


def test_neg(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(-data, -np_array)


def test_pow(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data ** 3, np_array ** 3)


def test_truediv(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data / 3, np_array / 3)


def test_sub(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(data - 3, np_array - 3)


def test_len(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert len(data) == len(np_array)


# index beyond array size (it will grow as needed)
def test_growing_array_by_index():
    data = DynamicArray(shape=(5, 2), index_expansion=True)
    for i in range(11):
        data[i, 1] = i
    answer = np.column_stack((np.zeros(11), np.linspace(0, 10, 11)))
    assert np.array_equal(data, answer)


def test_error_index_zero():
    data = DynamicArray(shape=(5, 2), index_expansion=True)
    data[8, 1] = 1
    answer = [[0, 0] for i in range(9)]
    answer[8][1] = 1
    assert np.array_equal(data, answer)


def test_growing_array_by_slice():
    data = DynamicArray(shape=(5, 2), index_expansion=True)
    a = [1, 2, 3]
    aa = [[i, i] for i in a]
    data[5:8] = aa
    answer_ = [0, 0, 0, 0, 0, 1, 2, 3]
    answer = [[i, i] for i in answer_]
    assert np.array_equal(data, answer)


# modify data with indexing
def test_indexing():
    data = DynamicArray(shape=(8, 2))
    data.append(np.linspace(0, 9, 10).reshape(5, 2))
    data[3] = [40, 40]
    assert all(data[3] == [40, 40])


def test_indexing_neg():
    data = DynamicArray(shape=(8, 2))
    data.append(np.linspace(0, 9, 10).reshape(5, 2))
    data[-1, 0] = 100
    assert data[4, 0] == 100


def test_indexing_slice():
    data = DynamicArray(shape=(8, 2))
    data.append(np.linspace(0, 13, 14).reshape(7, 2))
    data[3:6] = [[0, 0], [0, 0], [0, 0]]
    answer = [[0, 0] for i in range(7)]
    answer[0] = [0, 1]
    answer[1] = [2, 3]
    answer[2] = [4, 5]
    answer[6] = [12, 13]
    assert np.array_equal(data, answer)


# Errors
def test_error_index_get():
    with pytest.raises(IndexError):
        data = DynamicArray(shape=(8, 2))
        data.append(np.linspace(0, 13, 14).reshape(7, 2))
        print(data[100])


def test_error_index_set():
    with pytest.raises(IndexError):
        data = DynamicArray(shape=(8, 2))
        data.append(np.linspace(0, 13, 14).reshape(7, 2))
        data[100] = 100


def test_error_append_dict():
    with pytest.raises(ValueError):
        data = DynamicArray(shape=(8, 2))
        data.append({"fish": 1})


# numpy functions
def test_numpy_max(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert data.max() == np_array.max()


def test_numpy_item(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert data.item(2) == np_array.item(2)


def test_numpy_abs(np_array):
    data = DynamicArray(shape=(10, 2))
    data.append(np_array)
    assert np.array_equal(np.abs(data - 5), np.abs(np_array - 5))
