#!/usr/bin/env python
# coding=utf8
# File: lab_export.py

from rllab.utils import logger
___all__ = ['lab_export']

class lab_export(object):
    """decorator to export RLlab API"""

    def __init__(self, *args):
        self._name = args[0]

    def __call__(self, func):
        """Calls this decorator"""
        func._lab_api_names = self._name
        return func
