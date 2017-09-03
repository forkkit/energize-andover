#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:03:13 2017

@author: matt
"""

from new_logger import write_prop_values, init_csv, findNAE
import pandas as pd
from bacpypes.core import stop
from apscheduler.schedulers.blocking import BlockingScheduler


AHS_props = pd.DataFrame(
            [("Main (kW)",3007360),
             ("DHB (kW)",3017359),
             ("DG (kW)",3017523),
             ("DE (kW)",3017605),
             ("DL (kW)",3017769),
             ("M1 (kW)",3017441),
             ("AMDP (kW)",3017687),
             ("Main (kWh)",3007361),
             ("DHB (kWh)",3017360),
             ("DG (kWh)",3017524),
             ("DE (kWh)",3017606),
             ("DL (kWh)",3017770),
             ("M1 (kWh)",3017442),
             ("AMDP (kWh)",3017688)],
    columns=['Name', 'Identifier'])

s = BlockingScheduler()

findNAE()

@s.scheduled_job('cron', minute='*')
def task():
    write_prop_values(AHS_props)
    
def start():
    try:
        s.start()
    except(KeyboardInterrupt):
        s.shutdown()
        stop()