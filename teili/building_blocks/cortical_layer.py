# -*- coding: utf-8 -*-
""" This BuildingBlock class describes the organisation of a single 
cortical layer.
"""
# @Author: mmilde
# @Date:   2020-04-01 17:36:37

import sys
import time
import numpy as np

from brian2 import ms, mV, pA, SpikeGeneratorGroup,\
    SpikeMonitor, StateMonitor, core 