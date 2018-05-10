#!/usr/bin/env python
# coding=utf8
# File: setup.py
# Author: zhoubo_NLP(zhouboacmer@qq.com) 

import sys
import os
import re
from setuptools import setup
def _find_packages(prefix=''):
  packages = []
  path = '.' 
  prefix = prefix
  for root, _, files in os.walk(path):
    if '__init__.py' in files:
      packages.append(
        re.sub('^[^A-z0-9_]', '', root.replace('/', '.'))
      ) 
  return packages

setup(
    name='rllab',
    version=0.1,
    author="wangfan04,zhoubo01",
    packages=_find_packages(__name__),
    package_data={'': ['*.so']}
)
