from typing import Optional, Union, Tuple

import numpy as np


class DynamicArray:
    """
    A class to dynamically grow numpy array as data is added in an efficient manner.
    For arrays or column vectors (new rows added, but no new added columns).
    Example
    -------
    a = DynamicArray((100, 2))
    a.add(np.ones((20, 2)))
    a.add(np.ones((120, 2)))
    a.add(np.ones((10020, 2)))
    print(a.data)
    print(a.data.shape)
    """

    def __init__(self, shape: Optional[Union[int, Tuple]] = 100):
        self._data = np.empty(shape)
        self.capacity = self._data.shape[0]
        self.size = 0

    def add(self, x: np.ndarray):
        """ Add data to array. """
        if x.shape[0] > self.capacity:
            self._grow_capacity(x)

        self._data[self.size:self.size + x.shape[0]] = x
        self.size += x.shape[0]
        self.capacity -= x.shape[0]

    def _grow_capacity(self, x: np.ndarray):
        """ Grows the capacity of the _data array. """
        shape_ = list(self._data.shape)
        change_need = x.shape[0] - self.capacity
        if shape_[0] + self.capacity > change_need:
            # double in size
            self.capacity += shape_[0]
            shape_[0] = shape_[0] * 2
        else:
            # if doubling is not enough, grow to fit incoming data exactly.
            self.capacity += change_need
            shape_[0] = shape_[0] + change_need

        newdata = np.empty(shape_)
        newdata[:self._data.shape[0]] = self._data
        self._data = newdata

    @property
    def data(self) -> np.ndarray:
        """ Returns data without extra spaces. """
        return self._data[:self.size]