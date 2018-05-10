#!/usr/bin/env python
# coding=utf8
# File: argtools.py


import inspect
import functools32 as functools

__all__ = ['call_only_once']


_FUNC_CALLED = set()

def call_only_once(func):
    """
    Decorate a method or property of a class, so that this method can only
    be called once for every instance.
    Calling it more than once will result in exception.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # cannot use hasattr here, because hasattr tries to getattr, which
        # fails if func is a property
        assert func.__name__ in dir(self), "call_only_once can only be used on method or property!"

        cls = type(self)
        # cannot use ismethod(), because decorated method becomes a function
        is_method = inspect.isfunction(getattr(cls, func.__name__))
        key = (self, func)
        assert key not in _FUNC_CALLED, \
            "{} {}.{} can only be called once per object!".format(
                'Method' if is_method else 'Property',
                cls.__name__, func.__name__)
        _FUNC_CALLED.add(key)

        return func(*args, **kwargs)

    return wrapper
