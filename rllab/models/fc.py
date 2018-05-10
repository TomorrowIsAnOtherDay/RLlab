# -*- coding: utf-8 -*-
# File: fc.py


import paddle.fluid as fluid

from .common import layer_register, VariableHolder
from ..utils import logger
from paddle.fluid.param_attr import ParamAttr

__all__ = ['FullyConnected']


@layer_register(log_shape=True)
def FullyConnected(
        input,
        size,
        act=None,
        name=None,
        use_bias=True):
    """
    A wrapper around `tf.layers.Dense`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).
    Variable Names:
    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """

    param_attr = ParamAttr(name='{}_W'.format(name))
    bias_attr = ParamAttr(name='{}_b'.format(name))

    ret = fluid.layers.fc(
            input=input,
            size=size,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)
    #var_W = fluid.global_scope().find_var(param_attr.name).get_tensor()
    ret.variables = VariableHolder(W=param_attr.name)
    return ret

if __name__ == '__main__':
    import paddle.fluid as fluid
    x = fluid.layers.data(name='state', shape=[3], dtype='float32')
    fc1 = FullyConnected('policy/fc1', x, size=256, act='relu')
    logger.info("fc1:{}".format(fc1))

    vars = fluid.default_main_program().list_vars()
