#!/usr/bin/env python
# coding=utf8
# File: fs.py

import os
import errno
import tqdm
from . import logger

__all__ = ['mkdir_p']


def mkdir_p(dirname):
    """ Make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
