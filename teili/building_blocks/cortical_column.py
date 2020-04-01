# -*- coding: utf-8 -*-
""" This BuildingBlock class describes the organisation of a single 
cortical column.
"""
# @Author: mmilde
# @Date:   2020-04-01 17:36:45

import sys
import time
import numpy as np

from brian2 import ms, mV, pA, SpikeGeneratorGroup,\
    SpikeMonitor, StateMonitor, core 