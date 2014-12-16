#! /usr/bin/env python

"""Provide an interface to variational Bayes with pypmc"""
from __future__ import print_function, division

import eos
import hdf5_io

import vb
from make_analysis import make_analysis
import pypmc
from pypmc.mix_adapt.r_value import make_r_gaussmix
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.tools import plot_mixture
from pypmc.tools.parallel_sampler import MPISampler

import argparse
import h5py
import numpy as np
import os, sys

from mpi4py.MPI import COMM_WORLD
rank = COMM_WORLD.Get_rank()
master = (rank == 0)

hdf5_subdirectory = '/importance_samples'

class IS(object):
    def __init__(self, analysis, args):
        self.analysis = analysis
        self.dim = len(analysis.priors)

        # make sure that every process has a different random number generator seed
        np.random.seed(args.seed + rank)

        # parse mixture
        mixture = hdf5_io.read_mixture(args.input, args.primary_group + vb.hdf5_subdirectory)
#         from matplotlib import pyplot
#         plot_mixture(mixture)
#         pyplot.show()
#
        hyper = hdf5_io.read_vb_hyperparameters(args.input, args.primary_group + vb.hdf5_subdirectory)

        # same number of n_samples in each process
        self.n_samples = args.samples // COMM_WORLD.Get_size()

        # define the importance sampler
        SequentialIS = pypmc.sampler.importance_sampling.DeterministicIS
        self.parallel_sampler = MPISampler(SequentialIS, target=analysis, proposal=mixture, prealloc=self.n_samples)

        hdf5_io.save_analysis(args.output, '/', self.analysis)

    def run(self):
        self.parallel_sampler.run(self.n_samples)
        if master:
            hdf5_io.save_is_samples(args.output, args.primary_group + hdf5_subdirectory, self.parallel_sampler.history_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MCMC on EOS analysis from python")
    parser.add_argument("--analysis-from",
                        help="Specify where the `eos.Analysis` instance shall be read off. " \
                        "Either specify a python module (for example `module.analysis`) or `env` " \
                        "[default] for reading off the environement variables.",
                        type=str, default='env')
    parser.add_argument("--analysis-info", help='Print constraints, parameters, and observables',
                        type=int, default=False)
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
    if args.step is None:
        args.primary_group = ''
    else:
        args.primary_group = '/step_%d' % args.step

    ###
    # take action
    ###
    sampler = IS(analysis, args)
    sampler.run()
