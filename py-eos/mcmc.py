#! /usr/bin/env python

"""Provide an interface to mcmc sampling with pypmc"""
from __future__ import print_function, division

import eos
from make_analysis import make_analysis
import pypmc

import argparse
import h5py
import numpy as np

def save_analysis(file, analysis, intermediate=''):
    """Store analysis in human-readable format in hdf5 file.

    ``intermediate`` is a string to distinguish such as ``chain #1``
     if multiple chains are to be stored. A separate group
     ``/descriptions/$intermediate`` will be created.

    Example:
    $ h5ls -r file.hdf5
    /                        Group
    /descriptions            Group
    /descriptions/constraints Dataset {3}
    /descriptions/parameters Dataset {15}

    $ h5ls -d file.hdf5/descriptions/constraints
    constraints              Dataset {3}
    Data:
        (0) "B->K::f_0+f_++f_T@HPQCD-2013A",

    $ h5ls -d file.hdf5/descriptions/parameters
    parameters               Dataset {15}
    Data:
        (0) "Parameter: Re{cT}, prior type: flat, range: [-1,1], value = 0.981878",

    """

    desc  = 'descriptions'
    const = 'constraints'
    param = 'parameters'

    group = file.create_group(desc)
    if intermediate:
        group = file[desc].create_group(intermediate)

    # variable-length ASCII string
    dt = h5py.special_dtype(vlen=bytes)
    ds_const = group.create_dataset(const, (len(analysis.constraints),), dtype=dt)
    dt = np.dtype({'names': ['name', 'min', 'max', 'nuisance', 'info'],
                   'formats': [dt, np.float64, np.float64, np.uint8, dt]})
    ds_param = group.create_dataset(param, (len(analysis.priors),), dtype=dt)

    for i, c in enumerate(analysis.constraints):
        ds_const[i] = c.name

    ana_str = repr(analysis)

    # index where a line with parameter info starts
    line_start = ana_str.find('Parameter: ')
    i = 0
    while line_start != -1:
        end = ana_str.find(', value = ', line_start)
        line = ana_str[line_start:end]
        p = analysis.priors[i]
        ds_param[i] = (p.name, p.range_min, p.range_max, p.nuisance, line)
        i += 1
        # search forward in next line
        line_start = ana_str.find('Parameter: ', end)

    assert i == len(analysis.priors)

class MCMC_Sampler(object):
    def __init__(self, analysis, args):
        self.analysis = analysis
        self.dim = len(analysis.priors)

        # make sure that every process has a different random number generator seed
        np.random.seed(args.seed)

        # not too clean but fast prerun
        if args.eos_integration_points is not None:
            eos.set_integrate_n(args.eos_integration_points)

        # define a proposal for the initial Markov chain run as in eos
        covariance = np.eye(self.dim)
        for i, prior in enumerate(self.analysis.priors):
            covariance[i,i] = prior.variance()
            if not prior.nuisance or args.scale_nuisance:
                covariance[i,i] /= args.scale_reduction**2
        # optimal scaling for Gaussian target/proposal
        covariance *= 2.38**2 / self.dim
        if args.proposal == 'gauss':
            local_prop = pypmc.density.gauss.LocalGauss(covariance)
        elif args.proposal == 'cauchy':
            # dof = 1
            local_prop = pypmc.density.student_t.LocalStudentT(covariance, 1)

        # create indicator function
        lower = [p.range_min for p in self.analysis.priors]
        upper = [p.range_max for p in self.analysis.priors]
        ind = pypmc.tools.indicator.hyperrectangle(lower, upper)

        start = self.draw_uniform_in_support()
        # log_target(start) must not be -inf!
        while self.analysis(start) == -np.inf:
            start[:] = self.draw_uniform_in_support()

        self.chain = pypmc.sampler.markov_chain.AdaptiveMarkovChain(self.analysis, local_prop, start,
                                                                      indicator=ind, prealloc=args.samples)

        self.burn_in = args.burn_in

        self.samples = args.samples
        self.update = args.update_after

        self.file_name = args.output

        file = h5py.File(self.file_name, 'w')
        # one row per sample, store parameter values and log(posterior)
        self.sample_dset = '/samples/chain #0'
        file.create_dataset(self.sample_dset, (self.samples, self.dim), 'float64')

        save_analysis(file, self.analysis, intermediate='chain #0')
        file.close()

    def draw_uniform_in_support(self):
        ''' draw initial points'''
        sample = np.empty(self.dim)
        for d in range(self.dim):
            sample[d] = np.random.uniform(self.analysis.priors[d].range_min, self.analysis.priors[d].range_max)
        return sample

    def run(self):
        '''Run adaptive Markov chain.'''
        if self.burn_in:
            print("Burn in: %d samples" % self.burn_in)
            self.chain.run(self.burn_in)
            self.chain.clear()

        print("Run %d iterations with chunk size %d" % (self.samples, self.update))
        iterations, last_accept, total_accept = 0, 0, 0
        i = 1
        while iterations < self.samples:
            last_accept = self.chain.run(self.update)
            total_accept += last_accept
            last_accept_rate = last_accept / self.update
            total_accept_rate = total_accept / (iterations + self.update)
            print("Acceptance rate in chunk %d: %4.2f%%, in total:  %4.2f%%" %
                  (i, last_accept_rate * 100, total_accept_rate * 100))
            try:
                self.chain.adapt()
            except np.linalg.LinAlgError:
                print('WARNING: set off-diagonal covariance elements to zero')
                self.chain.proposal.update(np.diag(np.diag(self.chain.proposal.sigma)))

            self.save_samples(iterations)

            iterations += self.update
            i += 1
        print('MCMC finished')

    def save_samples(self, iterations):
        '''Save batch of samples. File is closed to ensure flush to disk.'''

        file = h5py.File(self.file_name, 'r+')
        file[self.sample_dset][iterations:iterations + self.update] = self.chain.history[:]
        file.close()
        self.chain.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MCMC on EOS analysis from python")
    parser.add_argument("--analysis-from",
                        help="Specify where the `eos.Analysis` instance shall be read off. " \
                        "Either specify a python module (for example `module.analysis`) or `env` " \
                        "[default] for reading off the environement variables.",
                        type=str, default='env')
    parser.add_argument("--analysis-info", type=bool, help='Print constraints, parameters, and observables',
                        default=True)
    parser.add_argument("--burn-in", type=int, const=0, default=0, nargs='?')
    parser.add_argument("--eos-integration-points", type=int)
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--proposal", choices=['gauss', 'cauchy'], default='gauss', const='gauss', nargs='?')
    parser.add_argument("--samples", type=int)
    parser.add_argument("--scale-nuisance", type=bool)
    parser.add_argument("--scale-reduction", type=float, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--update-after", type=int)

    args = parser.parse_args()

    ###
    # validate and use arguments
    ###
    ana = make_analysis(args.analysis_from)
    if args.analysis_info:
        print(ana)

    assert args.scale_reduction > 0

    ###
    # take action
    ###
    sampler = MCMC_Sampler(ana, args)
    sampler.run()
