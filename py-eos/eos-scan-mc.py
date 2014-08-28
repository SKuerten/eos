#! /usr/bin/env python

import eos
import numpy as np

class Parser(object):
    """Parse a command line designed for ``eos-scan-mc``.

    Only accept arguments related to constructing the ``eos::Analysis``.

    """
    def __init__(self, cmd_line):
        import argparse

        parser = argparse.ArgumentParser(description="EOS scan mc emulator")

        class ActionConstraint(argparse.Action):
            """Store constraints and options together in order."""
            def __call__(self, parser, namespace, values, option_string=None):
                if not 'ordered_args' in namespace:
                    setattr(namespace, 'ordered_args', [])
                previous = namespace.ordered_args
                previous.append((self.dest, values))
                setattr(namespace, 'ordered_args', previous)

        class ActionParameter(argparse.Action):
            """Store parameter and prior definitions together in order."""
            def __call__(self, parser, namespace, values, option_string=None):
                if not 'ordered_params' in namespace:
                    setattr(namespace, 'ordered_params', [])
                previous = namespace.ordered_params
                previous.append((self.dest, values))
                setattr(namespace, 'ordered_params', previous)

        parser.add_argument("--constraint", action=ActionConstraint)
        parser.add_argument("--global-option", action=ActionConstraint, nargs=2)
        parser.add_argument("--kinematics", action=ActionConstraint, nargs=2)
        parser.add_argument("--observable-prior", action=ActionConstraint, nargs=4)

        parser.add_argument("--nuisance", action=ActionParameter, nargs='*')
        parser.add_argument("--prior", action=ActionParameter, nargs='*')
        parser.add_argument("--scan", action=ActionParameter, nargs=3)

        args = parser.parse_args(cmd_line.split())

        if not hasattr(args, "ordered_args"):
            raise Exception("No constraints found!")
        if not hasattr(args, "ordered_params"):
            raise Exception("No parameters found!")

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
                name, mini, central, maxi = arg[1]
                constraints.append(eos.ManualConstraint(name, float(mini), float(central), float(maxi), 0, kinematics, **options))
                # reset to empty kinematics for next observable
                kinematics = {}
            else:
                raise Exception("Unknown constraint specification: " + str(arg))

        class PriorInfo(object):
            """Store info for next prior while parsing."""
            name = None
            mini, maxi = -np.inf, np.inf
            number = None
            n_sigmas = None
            nuisance = False

        priors = []
        pinfo = PriorInfo()
        # reimplement eos-scan-mc.cc
        for arg in args.ordered_params:
            if arg[0] in ("scan", "nuisance"):
                pinfo.nuisance = (arg[0] == "nuisance")
                pinfo.name = arg[1][0]
                if len(arg[1]) == 2:
                    pinfo.n_sigmas = float(arg[1][1])
                else:
                    pinfo.mini, pinfo.maxi = float(arg[1][1]), float(arg[1][2])
                    if len(arg[1]) == 4:
                        pinfo.n_sigmas = float(arg[1][3])
                    elif len(arg[1]) > 4:
                        raise Exception("Invalid prior specification: " + str(arg))
                if pinfo.n_sigmas is not None and (pinfo.n_sigmas < 0 or pinfo.n_sigmas > 10):
                    raise ValueError("Invalid n_sigma %d" % pinfo.n_sigmas)
            elif arg[0] == "prior":
                prior_type = arg[1][0]
                prior = None
                if prior_type in ("gaussian", "log-gamma"):
                    lower, central, upper = float(arg[1][1]), float(arg[1][2]), float(arg[1][3])
                    if pinfo.n_sigmas > 0:
                        pinfo.mini = max(pinfo.mini, central - pinfo.n_sigmas * (central - lower))
                        pinfo.maxi = min(pinfo.maxi, central + pinfo.n_sigmas * (upper - central))
                        call_args = (pinfo.name, pinfo.mini, pinfo.maxi, lower, central, upper)
                        call_kwargs = dict(nuisance=pinfo.nuisance)
                        if prior_type == "gaussian":
                            prior = eos.LogPrior.Gauss(*call_args, **call_kwargs)
                        else:
                            prior = eos.LogPrior.LogGamma(*call_args, **call_kwargs)
                elif prior_type == "flat":
                    if pinfo.n_sigmas > 0:
                        raise Exception("Can't specify number of sigmas for flat prior")
                    prior = eos.LogPrior.Flat(pinfo.name, pinfo.mini, pinfo.maxi, nuisance=pinfo.nuisance)
                else:
                    raise Exception("Unknown prior distribution: " + prior_type)

                if not prior:
                    raise Exception("Prior could not be set for " + str(arg))
                else:
                    priors.append(prior)
                    # reset name etc.
                    pinfo = PriorInfo()
            else:
                raise Exception("Unknown argument: " + str(arg))

            self._constraints = constraints
            self._priors = priors

    @property
    def constraints(self):
        """List of eos.Constraints found in the command line."""
        return self._constraints

    @property
    def priors(self):
        """List of eos.Prior found in the command line."""
        return self._priors

if __name__ == "__main__":
    # grab data defined in bash scripts
    scenario = "scIII"
    data = "posthep13"
    from os import environ

    cmd_line = environ["CONSTRAINTS_" + data] + environ["SCAN_" + scenario] + environ["NUISANCE_" + data]

    parser = Parser(cmd_line)
    ana = eos.Analysis(parser.constraints, parser.priors)

    mode = np.array([+0.19279, -0.71658, -0.62039, +0.40307, -3.69668, -4.57915, +0.80256, +0.22532, +0.06386, +0.36407, +1.26332, +4.21015, +0.22937, +0.39652, -4.57418, +1.05489, +0.82686, +0.26285, -0.14794, +0.24709, -0.34462, -0.03520, +0.00607, -0.20922, +0.89997, +0.96355, +0.82546, +1.15800, +0.29384, -2.62888, +0.00153, +0.81901, +0.33862, 0.35404])

    print ana

    # target = 605.8885328
    print ana(mode)
