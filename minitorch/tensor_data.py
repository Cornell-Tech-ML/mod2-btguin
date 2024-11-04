from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index: Index tuple of ints.
        strides: Tensor strides for each dimension.

    Returns:
    -------
        Position in storage.

    """
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Ensures that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: Ordinal position to convert.
        shape: Tensor shape.
        out_index: Return index corresponding to position.

    """
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` from `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case,
    `shape` may be larger or have more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index: Multidimensional index of the bigger tensor.
        big_shape: Tensor shape of the bigger tensor.
        shape: Tensor shape of the smaller tensor.
        out_index: Multidimensional index of the smaller tensor.

    Returns:
    -------
        None

    """
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1: First shape.
        shape2: Second shape.

    Returns:
    -------
        Broadcasted shape.

    Raises:
    ------
        IndexingError: If shapes cannot be broadcast together.

    """
    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
    return tuple(reversed(c_rev))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a given shape.

    Args:
    ----
        shape: The shape of the tensor.

    Returns:
    -------
        A tuple representing the strides.

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initialize a TensorData instance.

        Args:
        ----
            storage: The storage for tensor elements.
            shape: The shape of the tensor.
            strides: Optional strides for each dimension. If not provided, contiguous strides are used.

        Raises:
        ------
            IndexingError: If the length of strides does not match the length of shape.

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:
        """Convert the tensor storage to CUDA device memory."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e., outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            True if the tensor is contiguous, False otherwise.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a: First shape.
            shape_b: Second shape.

        Returns:
        -------
            Broadcasted shape.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert a user-provided index to a single-dimensional storage index.

        Args:
        ----
            index: An integer or a tuple representing the tensor index.

        Returns:
        -------
            The corresponding single-dimensional index in storage.

        Raises:
        ------
            IndexingError: If the index is out of range or has incorrect dimensions.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate all possible indices for the tensor.

        Yields
        ------
            Tuples representing each valid index in the tensor.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index from the tensor.

        Returns
        -------
            A tuple representing a randomly selected valid index.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieve the value at the specified index.

        Args:
        ----
            key: A tuple representing the index.

        Returns:
        -------
            The value at the given index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at the specified index.

        Args:
        ----
            key: A tuple representing the index.
            val: The value to set at the given index.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return the core tensor data as a tuple.

        Returns
        -------
            A tuple containing storage, shape, and strides.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            order: A permutation of the dimension indices.

        Returns:
        -------
            A new `TensorData` instance with permuted dimensions.

        Raises:
        ------
            AssertionError: If the provided order is not a valid permutation of dimensions.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        return TensorData(
            self._storage,
            tuple([self.shape[o] for o in order]),
            tuple([self._strides[o] for o in order]),
        )

    def to_string(self) -> str:
        """Convert the tensor data to a formatted string.

        Returns
        -------
            A string representation of the tensor.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
