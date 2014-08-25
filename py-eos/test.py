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
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ordered_args' in namespace:
            setattr(namespace, 'ordered_args', [])
        previous = namespace.ordered_args
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_args', previous)

class ActionKinematics(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ordered_kinematics' in namespace:
            setattr(namespace, 'ordered_kinematics', [])
        previous = namespace.ordered_kinematics
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_kinematics', previous)

parser = argparse.ArgumentParser(description="EOS scan mc emulator")
parser.add_argument("--constraint", action=ActionConstraint)
parser.add_argument("--global-option", nargs=2, action=ActionConstraint)
parser.add_argument("--kinematics", action=ActionKinematics, nargs=2)
parser.add_argument("--nuisance")
parser.add_argument("--observable-prior", nargs=4, action=ActionKinematics)
parser.add_argument("--prior")
parser.add_argument("--scan", nargs=3)

args = parser.parse_args(env_constraints.split())

print args.__dict__
options = {}
constraints = []
for oa in args.ordered_args:
    if oa[0] == 'global_option':
        options[oa[1][0]] = oa[1][1]
    if oa[0] == "constraint":
        constraints.append(eos.Constraint(oa[1], **options))

print args.ordered_kinematics
# for ok in args.ordered_kinematics:
#     print



