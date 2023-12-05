from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    x = vals[arg]
    vals_l = list(vals)
    vals_r = list(vals)
    vals_l[arg] = x - epsilon
    vals_r[arg] = x + epsilon
    y_l = f(*vals_l)
    y_r = f(*vals_r)
    return (y_r - y_l) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited: Dict[int, bool] = {}
    sorted_vars: List[Variable] = []

    def visit(var: Variable) -> None:
        if var.unique_id in visited:
            return
        visited[var.unique_id] = True
        if not var.is_constant():
            for parent in var.parents:
                visit(parent)
            sorted_vars.insert(0, var)

    visit(variable)
    return sorted_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variables
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_vars = topological_sort(variable)
    derivates = {var.unique_id: 0.0 for var in sorted_vars}
    derivates[variable.unique_id] = deriv
    for current_var in sorted_vars:
        current_deriv = derivates[current_var.unique_id]
        if current_var.is_leaf():
            current_var.accumulate_derivative(current_deriv)
        else:
            for parent, partial_derivative in current_var.chain_rule(current_deriv):
                if parent.is_constant():
                    continue
                if parent.unique_id not in derivates:   
                    derivates[parent.unique_id] = 0.0
                derivates[parent.unique_id] += partial_derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
