from typing import Tuple

import numpy as np
import numba
from numba import cuda

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

TPB = 32

def _tensor_conv1d_cuda(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, k_width = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides

    cache = cuda.shared.array((TPB, TPB), dtype=numba.float64)
    out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    out_pos = cuda.blockIdx.x
    pos_x = cuda.threadIdx.x
    pos_y = cuda.threadIdx.y

    if out_pos < out_size:
        to_index(out_pos, out_shape, out_index)
        batch_index = out_index[0]
        out_channel_index = out_index[1]
        out_width_index = out_index[2]

        in_channels_index = pos_x
        offset = pos_y

        if in_channels_index < in_channels and offset < k_width:
            input_val = 0.0
            weight_val = 0.0
            if reverse:
                k_width_index = -offset
            else:
                k_width_index = offset
            if out_width_index + k_width_index >= 0 and out_width_index + k_width_index < width:
                input_pos = batch_index * s1[0] + in_channels_index * s1[1] + (out_width_index + k_width_index) * s1[2]
                weight_pos = out_channel_index * s2[0] + in_channels_index * s2[1] + offset * s2[2]
                input_val = input[input_pos]
                weight_val = weight[weight_pos]
            cache[in_channels_index, k_width_index] = input_val * weight_val
            cuda.syncthreads()

            if in_channels_index == 0 and k_width_index == 0:
                value = 0.0
                for c in range(in_channels):
                    for wi in range(k_width):
                        value += cache[c, wi]
                out[out_pos] = value

    
tensor_conv1d_cuda = cuda.jit()(_tensor_conv1d_cuda)

# def _tensor_conv2d_cuda(
#     out: Tensor,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     input: Tensor,
#     input_shape: Shape,
#     input_strides: Strides,
#     weight: Tensor,
#     weight_shape: Shape,
#     weight_strides: Strides,
#     reverse: bool,
# ) -> None:

#     batch_, out_channels, _, _ = out_shape
#     batch, in_channels, height, width = input_shape
#     out_channels_, in_channels_, kh, kw = weight_shape

#     assert(
#         batch == batch_
#         and in_channels == in_channels_
#         and out_channels == out_channels_
#     )

#     s1 = input_strides
#     s2 = weight_strides



class Conv1dFunCuda(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        output = input.zeros((batch, out_channels, w))
        threadsperblock = (TPB, TPB)
        blockspergrid = output.size
        tensor_conv1d_cuda[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        threadsperblock = (TPB, TPB)
        blockspergrid = grad_weight.size
        tensor_conv1d_cuda[blockspergrid, threadsperblock](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        blockspergrid = grad_input.size
        tensor_conv1d_cuda[blockspergrid, threadsperblock](
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight

# class Conv2dFunCuda(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, h, w = input.shape
#         out_channels, in_channels2, kh, kw = weight.shape
#         assert in_channels == in_channels2
#         output = input.zeros((batch, out_channels, h, w))
#         threadsperblock = (4, 16, 16)
#         blockspergrid = output.size
#         tensor_conv2d_cuda[blockspergrid, threadsperblock](
#             *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
#         )
#         return output







    

