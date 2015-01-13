#! /usr/bin/env python

"""Provide an interface to variational Bayes with pypmc"""
from __future__ import print_function, division

import eos
import hdf5_io
from make_analysis import make_analysis
from pypmc.mix_adapt.r_value import make_r_gaussmix
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.tools import plot_mixture

import argparse
import h5py
import numpy as np
import os, sys
from argparse import ArgumentError

sys.path.append(os.path.realpath('../plot'))
import samplingOutput

hdf5_subdirectory = '/vb'

def read_chains(file_name, **kwargs):
    '''Read mcmc chains and return a list of arrays'''
    output = samplingOutput.EOS_PYPMC_MCMC(file_name, **kwargs)

    # TODO check if analysis agrees

    # number of chains
#     assert len(output.samples) % output.reduced_length == 0, "chains not of same size"
#     K = len(output.samples // output.reduced_length)
    chains = output.individual_chains()
    return chains

class VB(object):
    def __init__(self, analysis, args):
        self.analysis = analysis
        self.args = args
        if args.mcmc_input:
            self.init_mcmc(args)
        elif args.is_input:
            self.init_is(args)
        else:
            raise ArgumentError('No input file specified')

        if args.prune:
            self.prune = float(args.prune)
        else:
            # Stephan's rule of thumb, reduce components agressively
            self.prune = 0.5 * self.vb.N / self.vb.K

        hdf5_io.save_analysis(args.output, '/', self.analysis)

    def init_is(self, args):
        # parse samples and weights
        output = samplingOutput.EOS_PYPMC_IS(args.is_input, args.step)

        # use MCMC results as prior
        hyperpar = hdf5_io.read_vb_hyperparameters(args.is_input, args.primary_group + hdf5_subdirectory)
        hyperpar_prior = self.posterior2prior(hyperpar)

        # but ignore weights from MCMC
        if args.step == 0:
            hyperpar_prior['alpha0'][:] = 1e-5

        # now combine both dicts
        hyperpar.update(hyperpar_prior)

        initial_guess = hdf5_io.read_mixture(args.is_input, args.primary_group + hdf5_subdirectory)

        self.vb = GaussianInference(output.samples, components=len(hyperpar['alpha']), **hyperpar)

    def posterior2prior(self, hyperpar):
        '''Convert posterior hyperparameter values into priors.
        Important to make independent copies so values can be changed inside GaussianInference.'''
        return dict(alpha0=hyperpar['alpha'].copy(), beta0=hyperpar['beta'].copy(), m0=hyperpar['m'].copy(),
                    nu0=hyperpar['nu'].copy(), W0=hyperpar['W'].copy())

    def init_mcmc(self, args):
        print(args.mcmc_input)
        output = samplingOutput.EOS_PYPMC_MCMC(args.mcmc_input,
                                               chains=args.chains,
                                               skip_initial=args.skip_initial)

        K_g = args.components_per_group
        long_patches = make_r_gaussmix(output.individual_chains(),
                                       K_g=K_g, critical_r=args.R_value,
                                       indices=self.indices(args))
        print("K_g:", K_g)
        print('Found %d groups with a total of %d patches' %
              (len(long_patches) // K_g, len(long_patches)))

        if args.thin:
            samples = output.samples[::args.thin]
        else:
            samples = output.samples


        if args.init_method == 'random':
            initial_guess = args.init_method
        elif args.init_method == 'long-patches':
            initial_guess = long_patches
        else:
            raise ArgumentError('Invalid initialization method: "%s"' % args.init_method)

        self.vb = GaussianInference(samples, initial_guess=initial_guess, components=len(long_patches),
                                    W0=np.diag([1e20] * len(self.analysis.priors)))

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

        return indices

    def run(self):
        self.vb.run(verbose=True, prune=self.prune)
        # todo code dublication
        hdf5_io.save_mixture(self.args.output, args.primary_group + hdf5_subdirectory, self.vb.make_mixture())
        hdf5_io.save_vb_hyperparameters(self.args.output, args.primary_group + hdf5_subdirectory, self.vb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MCMC on EOS analysis from python")

    # todo move common arguments to a dict such that mcmc and is don't have to replicate
    parser.add_argument("--analysis-from",
                        help="Specify where the `eos.Analysis` instance shall be read off. " \
                        "Either specify a python module (for example `module.analysis`) or `env` " \
                        "[default] for reading off the environment variables.",
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
    parser.add_argument("--init-method", help="Method to initialize variational Bayes", const='random', nargs='?')
    parser.add_argument("--is-input", help='File name that has the input from importance sampling', const='', nargs='?')
    parser.add_argument("--mcmc-input", help='File name that has the Markov chains', const='', nargs='?')
    parser.add_argument("--output", const='', nargs='?',
                        help="Output file name. If left empty, the output is written to 'vb.hdf5'" \
                        " in the same directory as the input file."
                        "For the special name 'APPEND', the output is written to the input file.")
    parser.add_argument('--prune', help="Components responsible for less than this value are removed.",
                        default=None, const=None, nargs='?')
    parser.add_argument('--R-value', help="Critical R value used in grouping of chains", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--skip-initial', help="Allows to skip the first fraction of iterations", type=float, default=0.0)
    parser.add_argument('--thin', help='Select only every N-th sample from the chains to reduce autocorrelation',
                        type=int, default=0)
    parser.add_argument("--step", type=int, default=0, help='Select a step: used to identify input/output')

    args = parser.parse_args()

    ###
    # validate and use arguments
    ###
    ana = make_analysis(args.analysis_from)
    if args.analysis_info:
        print(ana)

    np.random.seed(args.seed)

    if args.step is None:
        args.primary_group = ''
    else:
        args.primary_group = '/step_%d' % args.step

    if args.mcmc_input:
        input = args.mcmc_input
    elif args.is_input:
        input = args.is_input
    else:
        raise ArgumentError('No input file specified')

    if not args.output:
        output_dir = os.path.split(input)[0]
        args.output = os.path.join(output_dir, 'vb.hdf5')
    elif args.output == 'APPEND':
        args.output = input

    ###
    # take action
    ###
    vb = VB(ana, args)
    vb.run()
