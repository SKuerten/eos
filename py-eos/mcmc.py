#! /usr/bin/env python

"""Provide an interface to mcmc sampling with pypmc"""
from __future__ import print_function, division

import eos
from make_analysis import make_analysis
import pypmc
import hdf5_io

import argparse
import h5py
import numpy as np

hdf5_subdirectory = '/chain #0'

class Target(object):
    """Transform parameters with a linear transformation so pypmc samples
     *uncorrelated* parameters but eos sees the correlated parameters."""
    def __init__(self, analysis, filename=None):

        self.analysis = analysis

        if filename:
            # matrix stored row wise in txt file
            covariance = np.loadtxt(filename)

            # header contains first and last parameter name
            with open(filename) as f:
                _, first_name, last_name = f.readline().split()

            # find index of parameter in analysis
            first_index, last_index = None, None
            for i, p in enumerate(analysis.priors):
                if p.name == first_name:
                    first_index = i
                elif p.name == last_name:
                    last_index = i

            def check(index, parname):
                assert index is not None, "Parameter '%s' from '%s' not found in current analysis" % (parname, filename)
            check(first_index, first_name)
            check(last_index, last_name)
            assert len(covariance) == last_index - first_index + 1

            self.slice = slice(first_index, last_index + 1)

            # Cholesky
            self.chol = np.linalg.cholesky(covariance)
            self.inv = np.linalg.inv(self.chol)

    def __call__(self, x):
        """x: pypmc parameters"""
        if hasattr(self, 'chol'):
            # correlate parameters
            y = x.copy()
            y[self.slice] = self.chol.dot(y[self.slice])
            # check if transformed values are valid, others are checked inside of `analysis`
            for yi, p in zip(y[self.slice], self.analysis.priors[self.slice]):
                if yi < p.range_min or yi > p.range_max:
                    return -np.inf
            return self.analysis(y)
        else:
            return self.analysis(x)

class MCMC_Sampler(object):
    def __init__(self, analysis, args):
        self.analysis = analysis
        self.target = Target(self.analysis, args.covariance)
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

        # transformation should make parameters approximately independent
        if hasattr(self.target, 'chol'):
            for i in range(self.target.slice.start, self.target.slice.stop):
                covariance[i,i] = 1

        # optimal scaling for Gaussian target/proposal
        covariance *= 2.38**2 / self.dim
        if args.proposal == 'gauss':
            local_prop = pypmc.density.gauss.LocalGauss(covariance)
        elif args.proposal == 'cauchy':
            # dof = 1
            local_prop = pypmc.density.student_t.LocalStudentT(covariance, 1)

        # create indicator function
        lower = np.array([p.range_min for p in self.analysis.priors])
        upper = np.array([p.range_max for p in self.analysis.priors])

        # we don't know boundary of transformed parameters explicitly
        # but we check inside target that original parameters are fine
        if hasattr(self.target, 'chol'):
            lower[self.target.slice] = -np.inf
            upper[self.target.slice] = +np.inf
        ind = pypmc.tools.indicator.hyperrectangle(lower, upper)

        # initial values of chain
        valid_args = ['uniform', 'fixed']
        msg = '--initial-values must begin with one of ' + str(valid_args) + ': ' + str(args.initial_values)
        if isinstance(args.initial_values, basestring):
            assert args.initial_values in valid_args, msg

            if args.initial_values == 'uniform':
                start = self.draw_uniform_in_support()
                # log_target(start) must not be -inf!
                while self.analysis(start) == -np.inf:
                    start[:] = self.draw_uniform_in_support()
            elif args.initial_values == 'fixed':
                start = self.fixed_initial_values(args)
        else:
            # expect list
            assert isinstance(args.initial_values, list)
            assert args.initial_values[0] in valid_args, msg

            if args.initial_values[0] == 'fixed':
                start = self.fixed_initial_values(args)

        print('initial values')
        for d in range(self.dim):
            print(d,":", self.analysis.priors[d].name, start[d])

        # uncorrelate initial values
        if hasattr(self.target, 'inv'):
            start[self.target.slice] = self.target.inv.dot(start[self.target.slice])

        # check if start exists and if it gives rise to a valid target value
        assert self.target(start) != -np.inf, 'Initial values %s invalid' % str(start)

        self.chain = pypmc.sampler.markov_chain.AdaptiveMarkovChain(self.target, local_prop, start,
                                                                    save_target_values=True,
                                                                    indicator=ind, prealloc=args.samples)
        self.chain.set_adapt_params(covar_scale_multiplier=float(args.scale_multiplier),
                                    force_acceptance_min=float(args.acceptance_min),
                                    force_acceptance_max=float(args.acceptance_max))
        print("initial scale factor", self.chain.covar_scale_factor)
        self.burn_in = args.burn_in

        print("initial std. devs.")
        self.initial_std_dev = np.sqrt(np.diag(self.chain.proposal.sigma))
        print(self.initial_std_dev)

        self.samples = args.samples
        self.update = args.update_after

        self.file_name = args.output

        with h5py.File(self.file_name, 'w') as file:
            # one row per sample, store parameter values and log(posterior)
            # in resizable data set
            kwargs = dict(dtype='float64', fletcher32=True, compression="gzip")
            self.sample_dset = hdf5_subdirectory + '/samples'
            file.create_dataset(self.sample_dset, shape=(0, self.dim), maxshape=(self.samples, self.dim), **kwargs)
            self.target_dset = hdf5_subdirectory + '/log_posterior'
            file.create_dataset(self.target_dset, shape=(0,), maxshape=(self.samples,), **kwargs)

        hdf5_io.save_analysis(self.file_name, hdf5_subdirectory, self.analysis)

    def draw_uniform_in_support(self):
        ''' draw initial points'''
        sample = np.empty(self.dim)
        for d in range(self.dim):
            sample[d] = np.random.uniform(self.analysis.priors[d].range_min, self.analysis.priors[d].range_max)
        return sample

    def fixed_initial_values(self, args):
        """Created fixed initial values, either from user given or default values."""
        # need some number of key-value pairs + tag at first position
        assert len(args.initial_values) % 2 == 1
        # key value pairs
        values = {}
        for i in range(len(args.initial_values[1:]) // 2):
            values[args.initial_values[1 + 2 * i]] = float(args.initial_values[1 + 2 * i + 1])

        # take value from user if given, else default from eos/parameters.cc
        start = np.empty(self.dim)
        for d in range(self.dim):
            name = self.analysis.priors[d].name
            start[d] = values.get(name, self.analysis[name])

        return start

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
        print("final/initial std. devs.")
        print(np.sqrt(np.diag(self.chain.proposal.sigma)) / self.initial_std_dev)
        print("final scale factor", self.chain.covar_scale_factor)

    def save_samples(self, iterations):
        '''Save batch of samples. File is closed to ensure flush to disk.'''

        with h5py.File(self.file_name, 'r+') as file:
            new_length = iterations + self.update

            # history has rank 2 but 2nd rank can be ignored
            file[self.target_dset].resize(new_length, axis=0)
            file[self.target_dset][iterations:new_length] = self.chain.target_values[:][:,0]

            file[self.sample_dset].resize(new_length, axis=0)

            # correlate parameters
            samples = self.chain.history[:]
            if hasattr(self.target, 'chol'):
                for s in samples:
                    s[self.target.slice] = self.target.chol.dot(s[self.target.slice])
            file[self.sample_dset][iterations:new_length] = samples

        self.chain.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MCMC on EOS analysis from python")
    parser.add_argument("--acceptance-min", type=float, const=0.15, default=0.15, nargs='?')
    parser.add_argument("--acceptance-max", type=float, const=0.35, default=0.35, nargs='?')
    parser.add_argument("--analysis-from",
                        help="Specify where the `eos.Analysis` instance shall be read off. " \
                        "Either specify a python module (for example `module.analysis`) or `env` " \
                        "[default] for reading off the environement variables.",
                        type=str, default='env')
    parser.add_argument("--analysis-info", type=bool, help='Print constraints, parameters, and observables',
                        default=True)
    parser.add_argument("--burn-in", type=int, const=0, default=0, nargs='?')
    parser.add_argument("--covariance", nargs='?', const=None,
                        help='''File name of covariance matrix to uncorrelate parameters for easier sampling.
                        Expect format
                        with header containing the name of the first and last parameter to transform.
                        The file should be readable by `numpy.loadtxt`
                        Example:
                        # first_par last_par
                        1.0 ....
                        .... 1.24...
                        ''')
    parser.add_argument("--eos-integration-points", type=int)
    parser.add_argument("--initial-values", default='uniform', nargs='*')
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--proposal", choices=['gauss', 'cauchy'], default='gauss', const='gauss', nargs='?')
    parser.add_argument("--samples", type=int)
    parser.add_argument("--scale-nuisance", type=bool)
    parser.add_argument("--scale-multiplier", type=float, default=1.5, const=1.5, nargs='?',
                        help='Scale factor to adjust covariance if outside of target acceptance window')
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
