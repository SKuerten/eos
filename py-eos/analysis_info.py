#! /usr/bin/env python

"""Print analysis in human-readable format"""
from __future__ import print_function

from make_analysis import make_analysis

import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print analysis in human-readable format")
    parser.add_argument("--analysis-from", help="Specify where the `eos.Analysis` instance shall be read off. Either specify a python module (for example `module.analysis`) or `env` (default) for reading off the environement variables.",
                        type=str, action='store', default='env')

    args = parser.parse_args()

    ana = make_analysis(args.analysis_from)
    print(ana)
