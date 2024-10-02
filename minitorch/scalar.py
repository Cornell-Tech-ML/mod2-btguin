from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference

from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)


ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """Stores the history of `Function` operations used to construct the current Variable.

    Attributes
    ----------
    last_fn : Optional[Type[ScalarFunction]]
        The last Function that was called.
    ctx : Optional[Context]
        The context for that Function.
    inputs : Sequence[Scalar]
        The inputs given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# Task 1.2 and 1.4: Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation tracking.

    Scalar Variables behave as close as possible to standard Python numbers while
    also tracking the operations that led to the number's creation. They can only
    be manipulated by `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `x` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: Value to be accumulated.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable was created by the user (no `last_fn`)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Retrieve the parent variables in the computational graph.

        Returns
        -------
            Iterable[Variable]: The input variables that are parents of the current scalar.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this Scalar's inputs.

        This method is used during the backward pass of autodifferentiation to
        propagate gradients through the computational graph.

        Args:
        ----
            d_output (Any): The gradient of the final output with respect to this Scalar.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A list of tuples, each containing an input
            Variable and its corresponding gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # ASSIGN1.3
        x = h.last_fn._backward(h.ctx, d_output)
        return list(zip(h.inputs, x))
        # END ASSIGN1.3

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                    (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __add__(self, b: ScalarLike) -> Scalar:
        # ASSIGN1.2
        return Add.apply(self, b)
        # END ASSIGN1.2

    def __lt__(self, b: ScalarLike) -> Scalar:
        # ASSIGN1.2
        return LT.apply(self, b)
        # END ASSIGN1.2

    def __gt__(self, b: ScalarLike) -> Scalar:
        # ASSIGN1.2
        return LT.apply(b, self)
        # END ASSIGN1.2

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        # ASSIGN1.2
        return EQ.apply(b, self)
        # END ASSIGN1.2

    def __sub__(self, b: ScalarLike) -> Scalar:
        # ASSIGN1.2
        return Add.apply(self, -b)
        # END ASSIGN1.2

    def __neg__(self) -> Scalar:
        # ASSIGN1.2
        return Neg.apply(self)
        # END ASSIGN1.2

    def log(self) -> Scalar:
        """Logarithm function"""
        # ASSIGN1.2
        return Log.apply(self)
        # END ASSIGN1.2

    def exp(self) -> Scalar:
        """Exponential function"""
        # ASSIGN1.2
        return Exp.apply(self)
        # END ASSIGN1.2

    def sigmoid(self) -> Scalar:
        """Sigmoid function"""
        # ASSIGN1.2
        return Sigmoid.apply(self)
        # END ASSIGN1.2

    def relu(self) -> Scalar:
        """ReLU function"""
        # ASSIGN1.2
        return ReLU.apply(self)
        # END ASSIGN1.2


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f : function from n-scalars to 1-scalar.
        *scalars : n input scalar values.

    """
    print(f"\nf{f}")
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
