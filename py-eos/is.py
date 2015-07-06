#! /usr/bin/env python

"""Provide an interface to variational Bayes with pypmc"""
from __future__ import print_function, division

import argparse
import h5py
import numpy as np
import os, sys

import eos
import hdf5_io
from hdf5_io import primary_group
import vb
from make_analysis import make_analysis
import pypmc
from pypmc.mix_adapt.r_value import make_r_gaussmix
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.sampler.importance_sampling import combine_weights, ImportanceSampler
from pypmc.tools import plot_mixture
from pypmc.tools.parallel_sampler import MPISampler

sys.path.append(os.path.realpath('../plot'))
import samplingOutput

from mpi4py.MPI import COMM_WORLD
rank = COMM_WORLD.Get_rank()
master = (rank == 0)

hdf5_subdirectory = '/importance_samples'

class IS(object):
    def __init__(self, analysis, args):
        self.analysis = analysis
        self.dim = len(analysis.priors)

        self.args = args

        # not too clean but fast prerun
        if args.eos_integration_points is not None:
            eos.set_integrate_n(args.eos_integration_points)

        # make sure that every process has a different random number generator seed
        np.random.seed(args.seed + rank)

        # parse mixture
        mixture = samplingOutput.EOS_PYPMC_IS.read_mixture(args.input, primary_group(args.step) + vb.hdf5_subdirectory)

        # same number of n_samples in each process
        self.n_samples = args.samples // COMM_WORLD.Get_size()

        # define the importance sampler
        self.parallel_sampler = MPISampler(ImportanceSampler, target=analysis, proposal=mixture, prealloc=self.n_samples)

        hdf5_io.save_analysis(args.output, '/', self.analysis)

    def run(self):
        self.parallel_sampler.run(self.n_samples)
        print("rank %d: sampling completed" % rank)
        if master:
            hdf5_io.save_is_samples(self.args.output, primary_group(self.args.step) + hdf5_subdirectory, self.parallel_sampler.history_list)
            # todo every process could do that, not just master
            if self.args.deterministic_mixture and self.args.step > 0:
                self.deterministic_weights()

    def deterministic_weights(self):
        '''load previous and current samples and recompute deterministic-mixture weights'''
        (samples, merged_samples), (weights, merged_weights), proposals = \
            hdf5_io.read_is_history(self.args.input, '/', last_step=self.args.step)
        mixture_weights = combine_weights(samples, weights, proposals)
        hdf5_io.save_combined_weights(self.args.output, primary_group(self.args.step), mixture_weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MCMC on EOS analysis from python")
    parser.add_argument("--analysis-from",
                        help="Specify where the `eos.Analysis` instance shall be read off. " \
                        "Either specify a python module (for example `module.analysis`) or `env` " \
                        "[default] for reading off the environment variables.",
                        type=str, default='env')
    parser.add_argument("--analysis-info", help='Print constraints, parameters, and observables',
                        type=int, default=False)
    parser.add_argument("--deterministic-mixture", help='Merge samples of all previous steps and recompute' \
                        'the weights according to the deterministic-mixture algorithm by Cornuet et al.',
                        type=int, default=1)
    parser.add_argument("--eos-integration-points", type=int)
    parser.add_argument("--input", help="Input file name that contains the proposal")
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--samples", type=int,
                        help='Total number of importance n_samples; i.e. for the combination of all processes')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step", type=int, default=None, const=None, nargs='?')

    args = parser.parse_args()

    ###
    # validate and use arguments
    ###
    analysis = make_analysis(args.analysis_from)
    if args.analysis_info:
        print(analysis)

    ###
    # take action
    ###
    sampler = IS(analysis, args)
    sampler.run()
