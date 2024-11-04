from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context


if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


# class ScalarFunction(ABC):
class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.

    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        Args:
        ----
            *vals: Input values of ScalarLike type.

        Returns:
        -------
            Scalar: The result of applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the addition operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Perform the backward pass of the addition operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients with respect to inputs a and b.

        """
        return d_output, d_output


class Log(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the natural logarithm operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: Input value.

        Returns:
        -------
            float: The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the natural logarithm operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to input a.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the multiplication operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: The product of a and b.

        """
        # ASSIGN1.2
        ctx.save_for_backward(a, b)
        c = a * b
        return c
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the multiplication operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b.

        """
        # ASSIGN1.4
        a, b = ctx.saved_values
        return b * d_output, a * d_output
        # END ASSIGN1.4


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the inverse operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: Input value.

        Returns:
        -------
            float: The inverse of a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the inverse operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to input a.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the negation operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: Input value.

        Returns:
        -------
            float: The negation of a.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the negation operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to input a.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the sigmoid operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: Input value.

        Returns:
        -------
            float: The sigmoid of a.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the sigmoid operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to input a.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the ReLU operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: Input value.

        Returns:
        -------
            float: The ReLU of a.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the ReLU operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to input a.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the exponential operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: Input value.

        Returns:
        -------
            float: The exponential of a.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the exponential operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to input a.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the less-than comparison operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: 1.0 if a is less than b, else 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the less-than comparison operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b (always 0.0).

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the equality comparison operation.

        Args:
        ----
            ctx: The context for saving values for backward pass.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            float: 1.0 if a equals b, else 0.0.

        """
        # ASSIGN1.2
        return 1.0 if a == b else 0.0
        # END ASSIGN1.2

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the equality comparison operation.

        Args:
        ----
            ctx: The context containing saved values from forward pass.
            d_output: The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b (always 0.0).

        """
        # ASSIGN1.4
        return 0.0, 0.0
        # END ASSIGN1.4
