#!/usr/bin/env python
# coding=utf8
# File: variable_scope.py


from contextlib import contextmanager
from collections import defaultdict
from rllab.utils.lab_export import lab_export
import paddle.fluid as fluid


__all__ = ['variable_scope', 'get_variable_scope']


_ScopeStack = list()

@lab_export('variable_scope')
@contextmanager
def variable_scope(name):
    _ScopeStack.append(name)
    yield

    del _ScopeStack[-1]

@lab_export('get_variable_scope')
def get_variable_scope():
    """
        Returns:
            list: the current variable scope 
            An variable scope is a (nested) name scope ``target/policy``
    """
    if len(_ScopeStack):
        return '/'.join(_ScopeStack)
    else:
        None
