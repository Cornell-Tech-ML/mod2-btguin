"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementaiton of a prelude of elemtary functions.


def mul(x: float, y: float) -> float:
    """Multiply two floating point numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two floating point numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a floating point number."""
    return -x


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is qual to y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is qual to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two floating point numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return True if x is close to y within a small tolerance."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of x."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute e raised to the power of x."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the natural logarithm."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the inverse of x."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the inverse function."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in a list.

    Args:
    ----
        fn: A function to apply to each element.

    Returns:
    -------
        A new list with the function applied to each element.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use map and neg to negate each element in ls"""
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a binary function to pairs of elements from two lists.

    Args:
    ----
        fn: A binary function to apply to pairs of elements.

    Returns:
    -------
        Function that takes two equally sized lists ls1 and ls2, produce a new list applying fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWidth` and `add`"""
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce.

    Args:
    ----
        fn: A binary function to apply to the elements.
        start: An iterable of floats.

    Returns:
    -------
        Function that takes a list ls of elements and computes the reduction

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using reduce and add"""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using reduce and mul"""
    return reduce(mul, 1.0)(ls)
