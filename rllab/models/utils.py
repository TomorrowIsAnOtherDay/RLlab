#!/usr/bin/env python
# coding=utf8
# File: utils.py


__all__ = ['VariableHolder']

import six
import paddle.fluid as fluid
from ..utils import logger
import numpy as np

class VariableHolder(object):
    """ A proxy to access variables defined in a layer. """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: {name:variable}
        """
        self._vars = {}
        for k, v in six.iteritems(kwargs):
            self._add_variable(k, v)

    def _add_variable(self, name, var):
        assert name not in self._vars
        self._vars[name] = var

    def __setattr__(self, name, var):
        if not name.startswith('_'):
            self._add_variable(name, var)
        else:
            # private attributes
            super(VariableHolder, self).__setattr__(name, var)

    def __getattr__(self, name):
        """
        get variable 
        Return: Numpy array
        """
        var_name = self._vars[name]
        try:
            var = fluid.global_scope().find_var(var_name).get_tensor()
        except AttributeError:
            logger.critical("Unable to fetch variable:{}".format(var_name))
            logger.critical("You should init program before fetching variables".format(var_name))
            import sys
            sys.exit(-1)

        return np.array(var)

    def all(self):
        """
        Returns:
            list of all variables
        """
        return list(six.itervalues(self._vars))
