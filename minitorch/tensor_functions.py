"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    """Base class for autodifferentiation Functions"""

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Wrapper for backward pass"""
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Wrapper for forward pass"""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for inverse"""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for inverse"""
        (t1,) = ctx.saved_values
        return grad_output * (-1 / (t1 * t1))


class Add(Function):
    """Addition function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for addition"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition"""
        return grad_output, grad_output


class All(Function):
    """All function"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[int]) -> Tensor:
        """Return 1 if all are true"""
        ctx.save_for_backward(a.shape, dim)
        if dim is not None:
            return a.f.mul_reduce(a, dim)
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for all function"""
        # The gradient of 'all' operation is zero since it's not differentiable
        t1_shape, dim = ctx.saved_values
        return zeros(t1_shape, backend=grad_output.backend)


class Mul(Function):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for multiplication"""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication"""
        t1, t2 = ctx.saved_values
        return grad_output * t2, grad_output * t1


class Sigmoid(Function):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for sigmoid"""
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid"""
        (out,) = ctx.saved_values
        return grad_output * out * (1 - out)


class ReLU(Function):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for ReLU"""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU"""
        (t1,) = ctx.saved_values
        grad_input = grad_output * (t1 > 0.0)
        return grad_input


class Log(Function):
    """Natural logarithm function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for log"""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for log"""
        (t1,) = ctx.saved_values
        return grad_output / t1


class Exp(Function):
    """Exponential function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for exp"""
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exp"""
        (out,) = ctx.saved_values
        return grad_output * out


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a tensor of ones of size `shape`"""
    return minitorch.Tensor.make(
        [1.0] * int(operators.prod(shape)), shape, backend=backend
    )


class Sum(Function):
    """Sum function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[int]) -> Tensor:
        """Forward pass for sum"""
        ctx.save_for_backward(t1.shape, dim)
        if dim is None:
            # Reduce over all dimensions
            t1 = t1.contiguous().view(int(operators.prod(t1.shape)))
            dim = 0
        out = t1.f.add_reduce(t1, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sum"""
        t1_shape, dim = ctx.saved_values
        if dim is None:
            # Expand grad_output to the original shape
            return grad_output * ones(t1_shape, backend=grad_output.backend)
        else:
            # Adjust grad_output shape and expand
            shape = list(t1_shape)
            shape[dim] = 1
            grad_output = grad_output.view(*shape)
            return grad_output * ones(t1_shape, backend=grad_output.backend)

    @classmethod
    def apply(cls, t1: Tensor, dim: Optional[int] = None) -> Tensor:
        """Custom apply method to handle dim as an int"""
        need_grad = t1.requires_grad()
        ctx = Context(not need_grad)
        raw_t1 = t1.detach()
        c = cls.forward(ctx, raw_t1, dim)
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, (t1,))
        return minitorch.Tensor(c._tensor, back, backend=t1.backend)


class LT(Function):
    """Less than comparison function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for less than"""
        # Comparison operation; not differentiable
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less than"""
        t1, t2 = ctx.saved_values
        zero_t1 = zeros(t1.shape, backend=t1.backend)
        zero_t2 = zeros(t2.shape, backend=t2.backend)
        return zero_t1, zero_t2


class EQ(Function):
    """Equality comparison function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for equality"""
        # Comparison operation; not differentiable
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality"""
        t1, t2 = ctx.saved_values
        zero_t1 = zeros(t1.shape, backend=t1.backend)
        zero_t2 = zeros(t2.shape, backend=t2.backend)
        return zero_t1, zero_t2


class IsClose(Function):
    """Is close comparison function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for is close"""
        # Not differentiable
        ctx.save_for_backward(t1, t2)
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for is close"""
        t1, t2 = ctx.saved_values
        zero_t1 = zeros(t1.shape, backend=t1.backend)
        zero_t2 = zeros(t2.shape, backend=t2.backend)
        return zero_t1, zero_t2


class Permute(Function):
    """Permute function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tuple[int, ...]) -> Tensor:
        """Forward pass for permute"""
        ctx.save_for_backward(order)
        return t1._new(t1._tensor.permute(*order))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for permute"""
        (order,) = ctx.saved_values
        reverse_order = [0] * len(order)
        for i, j in enumerate(order):
            reverse_order[j] = i
        return grad_output.permute(*reverse_order)

    @classmethod
    def apply(cls, t1: Tensor, order: Tuple[int, ...]) -> Tensor:
        """Custom apply method to handle order as a tuple"""
        need_grad = t1.requires_grad()
        ctx = Context(not need_grad)
        raw_t1 = t1.detach()
        c = cls.forward(ctx, raw_t1, order)
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, (t1,))
        return minitorch.Tensor(c._tensor, back, backend=t1.backend)


class View(Function):
    """View function"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for view"""
        ctx.save_for_backward(a.shape, shape.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Backward pass for view"""
        (original_shape, shape_tensor_shape) = ctx.saved_values
        grad_input = grad_output.view(*original_shape)
        return (grad_input,)

    @classmethod
    def apply(cls, a: Tensor, shape: Tensor) -> Tensor:
        """Custom apply method to handle shape as a Tensor"""
        need_grad = a.requires_grad()
        ctx = Context(not need_grad)
        raw_a = a.detach()
        c = cls.forward(ctx, raw_a, shape)
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, (a,))
        return minitorch.Tensor(c._tensor, back, backend=a.backend)


class Copy(Function):
    """Copy function"""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    """Matrix multiplication function"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        grad_t1 = grad_output.f.matrix_multiply(grad_output, transpose(t2))
        grad_t2 = grad_output.f.matrix_multiply(transpose(t1), grad_output)
        return grad_t1, grad_t2


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute central difference approximation of the gradient"""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference approximation"""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
