#! /usr/bin/env python

"""Provide an interface to optimize eos::Analysis with algorithms from the nlopt package via python."""

import argparse
import eos_scan_mc
import numpy as np
import nlopt

from os import environ

class NLOPT_Wrapper(object):
    def __init__(self, analysis):
        self.analysis = analysis
        self.calls = 0

    def __call__(self, x, grad):
        self.calls += 1
        if grad.size > 0:
            raise NotImplementedError
        return self.analysis(x)

nlopt_algorithms = dict(GN_DIRECT=nlopt.GN_DIRECT,
                        GN_DIRECT_L=nlopt.GN_DIRECT_L,
                        GN_DIRECT_L_RAND=nlopt.GN_DIRECT_L_RAND,
                        GN_DIRECT_NOSCAL=nlopt.GN_DIRECT_NOSCAL,
                        GN_DIRECT_L_NOSCAL=nlopt.GN_DIRECT_L_NOSCAL,
                        GN_DIRECT_L_RAND_NOSCAL=nlopt.GN_DIRECT_L_RAND_NOSCAL,
                        GN_ORIG_DIRECT=nlopt.GN_ORIG_DIRECT,
                        GN_ORIG_DIRECT_L=nlopt.GN_ORIG_DIRECT_L,
                        GD_STOGO=nlopt.GD_STOGO,
                        GD_STOGO_RAND=nlopt.GD_STOGO_RAND,
                        LD_LBFGS_NOCEDAL=nlopt.LD_LBFGS_NOCEDAL,
                        LD_LBFGS=nlopt.LD_LBFGS,
                        LN_PRAXIS=nlopt.LN_PRAXIS,
                        LD_VAR1=nlopt.LD_VAR1,
                        LD_VAR2=nlopt.LD_VAR2,
                        LD_TNEWTON=nlopt.LD_TNEWTON,
                        LD_TNEWTON_RESTART=nlopt.LD_TNEWTON_RESTART,
                        LD_TNEWTON_PRECOND=nlopt.LD_TNEWTON_PRECOND,
                        LD_TNEWTON_PRECOND_RESTART=nlopt.LD_TNEWTON_PRECOND_RESTART,
                        GN_CRS2_LM=nlopt.GN_CRS2_LM,
                        GN_MLSL=nlopt.GN_MLSL,
                        GD_MLSL=nlopt.GD_MLSL,
                        GN_MLSL_LDS=nlopt.GN_MLSL_LDS,
                        GD_MLSL_LDS=nlopt.GD_MLSL_LDS,
                        LD_MMA=nlopt.LD_MMA,
                        LN_COBYLA=nlopt.LN_COBYLA,
                        LN_NEWUOA=nlopt.LN_NEWUOA,
                        LN_NEWUOA_BOUND=nlopt.LN_NEWUOA_BOUND,
                        LN_NELDERMEAD=nlopt.LN_NELDERMEAD,
                        LN_SBPLX=nlopt.LN_SBPLX,
                        LN_AUGLAG=nlopt.LN_AUGLAG,
                        LD_AUGLAG=nlopt.LD_AUGLAG,
                        LN_AUGLAG_EQ=nlopt.LN_AUGLAG_EQ,
                        LD_AUGLAG_EQ=nlopt.LD_AUGLAG_EQ,
                        LN_BOBYQA=nlopt.LN_BOBYQA,
                        GN_ISRES=nlopt.GN_ISRES,
                        AUGLAG=nlopt.AUGLAG,
                        AUGLAG_EQ=nlopt.AUGLAG_EQ,
                        G_MLSL=nlopt.G_MLSL,
                        G_MLSL_LDS=nlopt.G_MLSL_LDS,
                        LD_SLSQP=nlopt.LD_SLSQP,
                        LD_CCSAQ=nlopt.LD_CCSAQ,
                        GN_ESCH=nlopt.GN_ESCH,
                        )

def make_opt(alg, local_algorithm=None):

    cmd_line = environ["EOS_constraints"] + environ["EOS_scan"] + environ["EOS_nuisance"]

    parser = eos_scan_mc.Parser(cmd_line)
    ana = eos_scan_mc.eos.Analysis(parser.constraints, parser.priors)

    target_density = NLOPT_Wrapper(ana)

    optimizers = [nlopt.opt(nlopt_algorithms[alg], len(parser.priors)),
                  nlopt.opt(nlopt_algorithms[local_algorithm], len(parser.priors)) if local_algorithm else None]

    for opt, arg in zip(optimizers, ("", "_local")):
        if not opt:
            continue
        opt.set_max_objective(target_density)

        bounds_low  = [prior.range_min for prior in parser.priors]
        bounds_high = [prior.range_max for prior in parser.priors]
        opt.set_lower_bounds(bounds_low)
        opt.set_upper_bounds(bounds_high)

        try:
            tol = float(environ["EOS_opt_tol" + arg])
        except KeyError:
            tol = 1e-14

        opt.set_ftol_abs(tol)
        opt.set_xtol_rel(np.sqrt(tol))

        try:
            maxeval = int(environ["EOS_opt_maxeval" + arg])
        except KeyError:
            maxeval = 3000
        opt.set_maxeval(maxeval)

    if local_algorithm is not None:
        optimizers[0].set_local_optimizer(optimizers[1])

    return target_density, optimizers[0], optimizers[1]

def print_opt(opt):
    return opt.get_algorithm_name() + " with xtol=%g, maxeval=%d," % (opt.get_xtol_rel(), opt.get_maxeval())

if __name__ == '__main__':
    np.set_printoptions(precision=8)

    parser = argparse.ArgumentParser(description="Optimize EOS analysis from python")
    parser.add_argument("--algorithm")
    parser.add_argument("--local-algorithm", nargs='?', const=None)

    args = parser.parse_args()

    target_density, opt, local_opt = make_opt(args.algorithm, local_algorithm=args.local_algorithm)

    print target_density.analysis

    mode = environ["EOS_mode"]
    mode = np.array([float(x) for x in mode.split()[1:-1]])

    print "Starting", print_opt(opt), " with f =", target_density.analysis(mode), "at"
    print mode

    xopt = opt.optimize(mode)
    fmax = opt.last_optimum_value()

    print " Found maximum value", fmax, " after", target_density.calls, "function calls at"
    print xopt

    if local_opt:
        print "Continuing with local algorithm", print_opt(local_opt)
        target_density.calls = 0
        local_xopt = local_opt.optimize(xopt)
        local_fmax = local_opt.last_optimum_value()
        print " Found maximum value", local_fmax, " after", target_density.calls, "function calls at"
        print local_xopt
