import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor

from .tensor_strategies import tensors


@pytest.mark.task4_1
def test_conv1d_simple() -> None:
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFunCuda.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    print(input, weight)
    minitorch.grad_check(minitorch.Conv1dFunCuda.apply, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv1dFunCuda.apply, input, weight)