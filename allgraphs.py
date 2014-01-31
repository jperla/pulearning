#!/usr/bin/env python
import sys

FULL_GRAPH = False
if 'full' in sys.argv:
    FULL_GRAPH = True

SUPPRESS_PLOT = True

execfile('reproduce_generated_2008.py')
execfile('tryout_posonly.py')
execfile('graph_roc.py')
execfile('graph_switched_roc.py')
