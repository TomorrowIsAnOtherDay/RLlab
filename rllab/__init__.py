#!/usr/bin/env python
# coding=utf8
# File: __init__.py
# Author: zhoubo_NLP(zhouboacmer@qq.com) 


from rllab.models import *
from rllab.utils import logger
import sys

import paddle.fluid as fluid
lab = fluid.layers

for module in list(sys.modules.values()):
    if (not module or not hasattr(module, '__name__') or
        'rllab' not in module.__name__):
        continue
    for module_contents_name in dir(module):
        attr = getattr(module, module_contents_name)
        if hasattr(attr, '__dict__') and '_lab_api_names' in attr.__dict__:
            name = attr._lab_api_names
            setattr(lab, name, attr)
