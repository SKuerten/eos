#! /usr/bin/env python

"""Provide an interface to variational Bayes with pypmc"""
from __future__ import print_function, division

import eos
from make_analysis import make_analysis
from pypmc.mix_adapt.r_value import make_r_gaussmix
from pypmc.mix_adapt.variational import GaussianInference

import argparse
import h5py
import numpy as np
import os, sys

sys.path.append(os.path.realpath('../plot'))
import samplingOutput

def read_chains(file_name, **kwargs):
    '''Read mcmc chains and return a list of arrays'''
    output = samplingOutput.EOS_PYPMC_MCMC(file_name, **kwargs)

    # TODO check if analysis agrees

    # number of chains
#     assert len(output.samples) % output.reduced_length == 0, "chains not of same size"
#     K = len(output.samples // output.reduced_length)
    chains = output.individual_chains()
    print(len(chains))
    print(len(chains[0]))
    return chains

class VB(object):
    def __init__(self, analysis, args):
        self.analysis = analysis
        output = samplingOutput.EOS_PYPMC_MCMC(args.mcmc_input,
                                               chains=args.chains,
                                               skip_initial=args.skip_initial)

        print("R value grouping")
        long_patches = make_r_gaussmix(output.individual_chains(),
                                       K_g=args.components_per_group,
                                       indices=self.indices(args))

        self.vb = GaussianInference(output.samples, initial_guess=long_patches, W0=np.diag([1e20] * len(analysis.priors)))

    def indices(self, args):
        '''Indices of parameters to compute R value for'''
        if args.indices:
            indices = args.indices
        elif args.group_nuisance:
            # all parameters
            indices = list(range(len(self.analysis.priors)))
        else:
            # find non-nuisance parameters
            indices = []
            for i, p in enumerate(self.analysis.priors):
                if not p.nuisance:
                    indices.append(i)

        print('R group parameter indices', indices)
        return indices

    def run(self):
        self.vb.run(verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MCMC on EOS analysis from python")
    parser.add_argument("--analysis-from",
                        help="Specify where the `eos.Analysis` instance shall be read off. " \
                        "Either specify a python module (for example `module.analysis`) or `env` " \
                        "[default] for reading off the environement variables.",
                        type=str, default='env')
    parser.add_argument("--analysis-info", help='Print constraints, parameters, and observables',
                        type=int, default=False)
    parser.add_argument('--chains', type=int, nargs='+', metavar=('chain0', 'chain1'),
                        help="Use only the specified chains for plotting, instead of all available chains. Example: --chains 0 2 5")
    parser.add_argument("--components-per-group", help='Number of components per group as determined by R value of chains',
                        type=int)
    parser.add_argument("--group-nuisance", help='Compute R value in grouping also for nuisance parameters',
                        type=int, default=0)
    parser.add_argument('--indices', help="Use only the specified parameters for R value grouping. Example: --indices 0 2 5",
                        type=int, nargs='+', metavar=('index0', 'index1'))
    parser.add_argument("--mcmc-input", help='File name that has the Markov chains', default='')
    parser.add_argument('--skip-initial', help="Allows to skip the first fraction of iterations", type=float, default=0.0)

    print(sys.argv)
    args = parser.parse_args()

    ###
    # validate and use arguments
    ###
    ana = make_analysis(args.analysis_from)
    print(args)
    if args.analysis_info:
        print(ana)

    ###
    # take action
    ###
    vb = VB(ana, args)
    vb.run()
