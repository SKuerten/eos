#! /usr/bin/env python

"""Provide an interface to optimize eos::Analysis with algorithms from the nlopt package via python."""

import argparse
import eos_scan_mc
from os import environ
import numpy as np
import nlopt

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

def make_analysis():
    if args.analysis_from == "env":
        cmd_line = environ["EOS_SCAN"] + environ["EOS_NUISANCE"] + environ["EOS_CONSTRAINTS"]
        parser = eos_scan_mc.Parser(cmd_line)
        return eos_scan_mc.eos.Analysis(parser.constraints, parser.priors)
    else:
        module_hierarchy = args.analysis_from.split('.')
        module_to_import = str.join('.', module_hierarchy[:-1])
        analysis_name = module_hierarchy[-1]
        exec("from " + module_to_import + " import " + analysis_name + " as analysis")
        return analysis

def make_opt(ana, alg, tol, maxeval):
    priors = target_density.analysis.priors
    opt = nlopt.opt(nlopt_algorithms[alg], len(priors))

    opt.set_max_objective(target_density)

    bounds_low  = [prior.range_min for prior in priors]
    bounds_high = [prior.range_max for prior in priors]
    opt.set_lower_bounds(bounds_low)
    opt.set_upper_bounds(bounds_high)

    # convergence criteria
    opt.set_ftol_abs(tol)
    opt.set_xtol_rel(np.sqrt(tol))
    opt.set_maxeval(maxeval)

    return opt

def print_opt(opt):
    return opt.get_algorithm_name() + " with ftol=%g, maxeval=%d," % (opt.get_ftol_rel(), opt.get_maxeval())

if __name__ == '__main__':
    # avoid scientific notation as argparse has trouble with it
    np.set_printoptions(formatter={'float': lambda x: '%+.16f' % x})

    parser = argparse.ArgumentParser(description="Optimize EOS analysis from python")
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--analysis-from", help="Specify where the `eos.Analysis` instance shall be read off. Either specify a python module (for example `module.analysis`) or `env` (default) for reading off the environement variables.",
                        type=str, action='store', default='env')
    parser.add_argument("--initial-guess", nargs='*', help="Vector to seed the optimization (minus signs must be replaced by `n` if notation `0.3e-3` is used). Ex: { 0.1 n0.3e-3 -0.5 }")
    parser.add_argument("--local-algorithm", nargs='?', default=None)
    parser.add_argument("--max-evaluations", type=int, action='store')
    parser.add_argument("--max-evaluations-local", type=int, action='store')
    parser.add_argument("--tolerance", type=float, action='store', default=1e-14, help="Relative tolerance to declare convergence")
    parser.add_argument("--tolerance-local", type=float, default=1e-14, help="Relative tolerance to declare convergence for the local algorithm")

    # validate arguments
    args = parser.parse_args()
    assert len(args.initial_guess) > 2, "invalid specification of the initial guess:" + str(args.initial_guess)

    # now do something with them
    ana = make_analysis()
    target_density = NLOPT_Wrapper(ana)

    opt = make_opt(target_density, args.algorithm, maxeval=args.max_evaluations, tol=args.tolerance)

    if args.local_algorithm:
        local_opt = make_opt(target_density, args.local_algorithm, maxeval=args.max_evaluations_local, tol=args.tolerance_local)
        opt.set_local_optimizer(local_opt)

    print target_density.analysis

    start = np.array([float(x.replace('n','-')) for x in args.initial_guess[1:-1]])
    print "Starting", print_opt(opt), " with f =", target_density.analysis(start), "at"
    print start

    xopt = opt.optimize(start)
    fmax = opt.last_optimum_value()

    print " Found maximum value", fmax, " after", target_density.calls, "function calls at"
    print xopt

    if args.local_algorithm:
        print "Continuing with local algorithm", print_opt(local_opt)
        target_density.calls = 0
        local_xopt = local_opt.optimize(xopt)
        local_fmax = local_opt.last_optimum_value()
        print " Found maximum value", local_fmax, " after", target_density.calls, "function calls at"
        print local_xopt
