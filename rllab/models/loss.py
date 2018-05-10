#!/usr/bin/env python
# coding=utf8
# File: loss.py


import paddle.fluid as fluid

from .common import layer_register
from ..utils import logger
from paddle.fluid.param_attr import ParamAttr

__all__ = ['L2Loss', 'SquareError']


@layer_register(log_shape=False, use_scope=False)
def L2Loss(
        input,
        label,
        name=None):
    """
    A wrapper around `fluid.layers.square_error_cost`.
    """
    ret = fluid.layers.square_error_cost(
            input=input,
            label=label)
    return ret

SquareError = L2Loss


if __name__ == '__main__':
    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[3], dtype='float32')
    y = fluid.layers.data(name='y', shape=[3], dtype='float32')
    error = SquareError("square_error", x, y)
