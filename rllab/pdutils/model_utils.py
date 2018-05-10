#!/usr/bin/env python
# coding=utf8
# File: model_utils.py


from ..utils import logger
import paddle.fluid as fluid

__all__ = []

def get_shape_str(tensor):
    """
    Args:
        a tensor
    Returns:
        str: a string to describe the shape
    """
    assert isinstance(tensor, (fluid.framework.Variable)), "Not a tensor: {}".format(type(tensor))
    shape = map(int, list(tensor.shape))
    shape[0] = None if shape[0] == -1 else shape[0]
    shape_str = str(shape)
    return shape_str

if __name__ == '__main__':
    import paddle.fluid as fluid
    x = fluid.layers.data(name='state', shape=[3], dtype='float32')
    fc1 = fluid.layers.fc(input=x, size=256, act='relu')
    logger.info("[get_shape_str]:{}".format(get_shape_str(fc1)))
    logger.info("fc1:{}".format(fc1))
