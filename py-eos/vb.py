#! /usr/bin/env python

"""Provide an interface to variational Bayes with pypmc"""
from __future__ import print_function, division

import eos
import hdf5_io
from hdf5_io import primary_group
from make_analysis import make_analysis
from pypmc.mix_adapt.r_value import make_r_gaussmix
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.sampler.importance_sampling import combine_weights
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
            self._init_mcmc()
        elif args.is_input:
            self._init_is()
        else:
            raise ArgumentError('No input file specified')

        if args.prune:
            self.prune = float(args.prune)
        else:
            # Stephan's rule of thumb, reduce components agressively
            self.prune = 0.5 * self.vb.N / self.vb.K

        hdf5_io.save_analysis(args.output, '/', self.analysis)

    def _init_is(self):
        deterministic = self.args.deterministic_mixture and self.args.step > 1
        # parse samples and weights
        output = samplingOutput.EOS_PYPMC_IS(self.args.is_input, self.args.step,
                                             deterministic_mixture=deterministic)

        # use MCMC results as prior
        hyperpar = hdf5_io.read_vb_hyperparameters(self.args.is_input, primary_group(self.args.step) + hdf5_subdirectory)
        hyperpar_prior = self._posterior2prior(hyperpar)

        if deterministic:
            (samples, merged_samples), (weights, merged_weights), proposals = \
                hdf5_io.read_is_history(self.args.is_input, '/', last_step=self.args.step)
            mixture_weights = combine_weights(samples, weights, proposals)
            # weights (N,1) matrix => take only first column
            self.vb = GaussianInference(merged_samples[:], weights=merged_weights[:].T[0],
                                        components=len(hyperpar['alpha']), **hyperpar)

        else:
            # but ignore weights from MCMC
            if self.args.step == 0:
                hyperpar_prior['alpha0'][:] = 1e-5

            # now combine both dicts
            hyperpar.update(hyperpar_prior)

            self.vb = GaussianInference(output.samples, weights=output.weights,
                                        components=len(hyperpar['alpha']), **hyperpar)

        # now save results in next step
        self.args.step += 1

    def _init_mcmc(self):
        output = samplingOutput.EOS_PYPMC_MCMC(self.args.mcmc_input,
                                               chains=self.args.chains,
                                               skip_initial=self.args.skip_initial)
        K_g = self.args.components_per_group
        print("K_g:", K_g)

        if self.args.init_method == 'random':
            initial_guess = self.args.init_method
        elif self.args.init_method == 'long-patches':
            long_patches = make_r_gaussmix(output.individual_chains(),
                                           K_g=K_g, critical_r=self.args.R_value,
                                           indices=self._indices())
            print('Found %d groups with a total of %d patches' %
                  (len(long_patches) // K_g, len(long_patches)))
            initial_guess = long_patches
        else:
            raise ArgumentError('Invalid initialization method: "%s"' % self.args.init_method)

        if self.args.thin:
            samples = output.samples[::self.args.thin]
        else:
            samples = output.samples

        self.vb = GaussianInference(samples, initial_guess=initial_guess, components=K_g,
                                    W0=np.diag([1e20] * len(self.analysis.priors)))

    def _indices(self):
        '''Indices of parameters to compute R value for'''
        if self.args.indices:
            indices = self.args.indices
        elif self.args.group_nuisance:
            # all parameters
            indices = list(range(len(self.analysis.priors)))
        else:
            # find non-nuisance parameters
            indices = []
            for i, p in enumerate(self.analysis.priors):
                if not p.nuisance:
                    indices.append(i)

        return indices

    def _posterior2prior(self, hyperpar):
        '''Convert posterior hyperparameter values into priors.
        Important to make independent copies so values can be changed inside GaussianInference.'''
        return dict(alpha0=hyperpar['alpha'].copy(), beta0=hyperpar['beta'].copy(), m0=hyperpar['m'].copy(),
                    nu0=hyperpar['nu'].copy(), W0=hyperpar['W'].copy())

    def run(self):
        output_directory = primary_group(self.args.step) + hdf5_subdirectory

        if hdf5_io.exists_mixture(self.args.output, output_directory):
            raise KeyError('Output mixture exists already in directory "%s" in file "%s"' % \
                               (output_directory, self.args.output))

        kwargs = dict(prune=self.prune, verbose=True)
        if self.args.rel_tol is not None:
            kwargs['rel_tol'] = self.args.rel_tol
        print('kwargs:')
        print(kwargs)
        self.vb.run(**kwargs)

        # todo code dublication
        mix = self.vb.make_mixture()
        hdf5_io.save_mixture(self.args.output, output_directory, mix)
        hdf5_io.save_vb_hyperparameters(self.args.output, output_directory, self.vb)
        # plot_mixture(mix, 0, 2)
        # from matplotlib import pyplot as plt
        # plt.savefig('/tmp/vb.pdf')

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
    parser.add_argument("--deterministic-mixture", help='Merge samples of all previous steps and recompute' \
                        'the weights according to the deterministic-mixture algorithm by Cornuet et al.',
                        type=int, default=1)
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
    parser.add_argument('--rel-tol', type=float, default=None, const=None, nargs='?',
                        help="Relative tolerance to determine convergence of variational Bayes")
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
