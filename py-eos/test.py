#! /usr/bin/env python

from os import environ

env_constraints = environ["CONSTRAINTS_posthep13"]

# def get_option(line):
#     expr = "--global-option "
#     i = s.find(expr)
#     if i != -1:
#          s[i + len(expr) + 1:]

# constraints = []
# global_options = {}
#
# for s in env_constraints.splitlines():
#     # only one option to eos-scan-mc per line, starts with "--"
#     if not '--' in s:
#         continue
#     print s
#     expr = "--global-option model"
#     i = s.find(expr)
#     if i != -1:
#         print s[i + len(expr) + 1:]

###
# preprocessing:
# 1. global-option -> local option (form factors)
###

# env_constraints = env_constraints.replace("--global-option form-factors ", "--local-option form-factors ")
# env_constraints = env_constraints.replace("--global-option form-factors ", "--local-option form-factors ")

for s in env_constraints.splitlines():
    print s

import argparse
import eos

class ActionConstraint(argparse.Action):
    """Store constraints together in order"""
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ordered_args' in namespace:
            setattr(namespace, 'ordered_args', [])
        previous = namespace.ordered_args
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_args', previous)

parser = argparse.ArgumentParser(description="EOS scan mc emulator")
parser.add_argument("--constraint", action=ActionConstraint)
parser.add_argument("--global-option", nargs=2, action=ActionConstraint)
parser.add_argument("--kinematics", action=ActionConstraint, nargs=2)
parser.add_argument("--observable-prior", nargs=4, action=ActionConstraint)
parser.add_argument("--nuisance")
parser.add_argument("--prior")
parser.add_argument("--scan", nargs=3)

args = parser.parse_args(env_constraints.split())

options = {}
constraints = []
kinematics = {}

for arg in args.ordered_args:
    if arg[0] == 'global_option':
        # ('global_option', ['model', 'WilsonScan'])
        options[arg[1][0]] = arg[1][1]
    elif arg[0] == "constraint":
        # ('constraint', "B^0->K^*0mu^+mu^-::P'_5[14.18,16.00]@LHCb-2013")
        constraints.append(eos.Constraint(arg[1], **options))
    elif arg[0] == "kinematics":
        # ('kinematics', ['s', '0.0'])
        kinematics[arg[1][0]] = float(arg[1][1])
    elif arg[0] == 'observable_prior':
        # ('observable_prior', ['B->K^*::V(s)/A_1(s)', '0.93', '1.33', '1.73'])
        name, min, central, max = arg[1]
        constraints.append(eos.ManualConstraint(name, float(min), float(central), float(max), 0, kinematics, **options))
        # reset to empty kinematics for next observable
        kinematics = {}
    else:
        raise UnknownArgument(arg)

for c in constraints:
    print c




