"""
Provide consistent interface to parsing sampling output from various algorithms, and in various formats.

"""

import priors as priorDistributions

import h5py
import numpy as np
import os
import sys

# todo rename to description
class ParameterDefinition(object):
    """Properties of a fit parameter"""

    def __init__(self, name, min, max, nuisance=False, discrete=False, index=None):
        self.name = name
        self.min = min
        self.max = max
        self.nuisance = nuisance
        self.discrete = discrete
        self.range = (min, max)
        self.i = index

    def __repr__(self):
        return "{Name: %s, min: %g, max: %g, nuisance: %d, discrete: %d}" % \
                ( self.name, self.min, self.max, self.nuisance, self.discrete)

class SamplingOutput(object):
    """
    Base class of all sampling algorithms' outputs.

    """

    def __init__(self, *args, **kwargs):
        self.input_file_name = args[0]

        # indices of samples that are selected
        self.select = kwargs.get('select', [None, None])

        self._read(*args, **kwargs)

        # some members are mandatory
        if not hasattr(self, 'samples'):
            raise NotImplementedError("samples undefined")
        if not hasattr(self, 'weights'):
            raise NotImplementedError("weights undefined")
        if not hasattr(self, 'par_defs'):
            raise NotImplementedError("parameter definitions undefined")
        if not hasattr(self, 'priors'):
            raise NotImplementedError("priors undefined")

    def _read(self):
        """
        Read the data from the file

        """
        NotImplementedError()

    @classmethod
    def open(cls, *args, **kwargs):
        """Traverse concrete implementations to find one that can open the file successfully"""

        if not os.path.exists(args[0]):
            raise IOError("Could not open " + args[0])

        out = None
        names = []
        for c in cls.__subclasses__():
            names.append(c.__name__)
            try:
                out = c(*args, **kwargs)
                return out
            except:
                continue

        raise IOError("Could not open " + args[0] + " with any of %s" % str(names))

    def npar(self):
        """Return number of parameters"""
        return len(self.par_defs)

    def get(self, attribute):
        """General attribute getter"""
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            return None

    def filter(self, cuts):
        """
        Return indices of samples that pass cuts.

        Cuts are extracted from a dict in which
        key=parameter index,
        value=(min, max)

        """
        if not cuts:
            return

        # samples inside of integration region
        rows = range(len(self.samples))

        # apply each cut
        print("Using the following cuts to define the integration region: " % cuts)
        for par, cut in cuts.iteritems():
            print("--par: %s, min: %g, max: %g" % (self.par_defs[par].name, cut[0], cut[1]))

         # first apply lower, then upper cuts
        rows1 = np.logical_and.reduce([self.samples[:, par] >= cut[0] for par, cut in cuts.iteritems()])
        rows2 = np.logical_and.reduce([self.samples[:, par] <= cut[1] for par, cut in cuts.iteritems()])
        rows = np.logical_and(rows1, rows2)

        return rows

def read_descriptions(file, data_set, npar=None, samples=None):
    """Read parameter descriptions from an HDF5 file

    Required arguments:
    file -- the h5py HDF5 file object
    data_set -- the name of the data set containing the descriptions

    Keyword arguments:
    npar -- the number of parameters (guide if it can't be extracted from the file)
    samples -- the array of samples, one per row (needed only if ranges can't be extracted from the file)

    """
    # read out parameter info
    par_defs = []
    priors = []
    try:
        descriptions = file[data_set][:]
    # plain output may not include parameter information
    except KeyError:
        for i in range(npar):
            priors.append(None)
            par_defs.append(ParameterDefinition('par%d' % i,
                                                np.min(samples.T[i]),
                                                np.max(samples.T[i])))
    # output from an Analysis includes parameters and their priors
    else:
        f = priorDistributions.PriorFactory()
        for row in descriptions:
            par_defs.append(ParameterDefinition(row[0], row[1], row[2], row[3], False))

            try:
                prior_name, prior = f.create(row[4])
                assert(prior_name == row[0])
            except KeyError as e:
                prior = None
                print('Warning: in constructing prior for %s: %s' % (row[0], e.message))
            # if it not an analysis, there is no prior
            except IndexError:
                prior = None

            priors.append(prior)

    return par_defs, priors

class MCMC_Output(SamplingOutput):

    def _read(self, *args, **kwargs):
        self.chains = kwargs.get('chains', None)
        self.prerun = kwargs.get('prerun', True)
        self.skip_initial = kwargs.get('skip_initial', 0.2)

        self.single_chain = None
        if hasattr(self.chains, '__len__') and len(self.chains) == 1:
            self.single_chain = str(self.chains[0])

        print("Opening " + self.input_file_name)
        hdf5_file = h5py.File(self.input_file_name, 'r')

        n_chains_parsed = 0

        prefix = 'main run'
        if self.prerun:
            prefix = "prerun"

        # select all chains
        if self.chains:
            chains = self.chains
        else:
            chains = range(len(list(hdf5_file[prefix])))

        first_chain = str(chains[0])
        self.n_chains = len(chains)

        #read data
        full_length = len(hdf5_file[prefix + '/chain #' + first_chain + "/samples"])

        #adjust which range is drawn, default: full range
        if self.select[0] is  None and self.skip_initial is not None:
            self.select[0] = int(self.skip_initial * full_length)

        merged_chains = hdf5_file[prefix + '/chain #' + first_chain + "/samples"][self.select[0]:self.select[1]]
        n_chains_parsed += 1

        #save shape info
        self.chain_length = len(merged_chains)

        par_defs, priors = read_descriptions(hdf5_file,
                                             data_set='descriptions/' + prefix + '/chain #' + first_chain + "/parameters",
                                             npar=merged_chains.shape[1] - 1,
                                             samples=merged_chains)

        # read out mode from stats: always last row
        stats = hdf5_file[prefix + '/chain #' + first_chain + "/stats/mode"]
        modes = [stats[-1]]

        # read all remaining chains
        for chain in chains[1:]:
            data = hdf5_file[prefix + '/chain #%d/samples' % chain][self.select[0]:self.select[1]]
            merged_chains = np.concatenate((merged_chains, data), axis=0)
            modes.append(hdf5_file[prefix + '/chain #%d/stats/mode' % chain][-1])
            n_chains_parsed += 1

        hdf5_file.close()

        ###
        # assign members
        ###

        # all weights equal
        self.weights = np.ones(len(merged_chains))
        self.samples = merged_chains
        self.par_defs = par_defs
        self.priors = priors

        self._modes = np.array(modes)

        self.extract_chain_modes()

    def extract_chain_modes(self):
        """
        Find the mode of each chain and display it, ignoring the nuisance parameters.

        Assumes that each chain of same length.
        """
        print("There are %d chains, with a merged shape of %s" % (self.n_chains, self.samples.shape))
        for i in range(self.n_chains):
            mode = []
            if len(self._modes) > 0:
                max = self._modes[i][-1]
                for j in range(self.npar()):
                    mode.append(self._modes[i][j])
            else:
                index = np.argmax(self.out.samples().T[-1][i * self.chain_length : (i + 1) * self.chain_length])
                max = None
                for j in range(self.out.npar()):
                    if self.out.par_defs()[j].nuisance and not self.use_nuisance:
                        continue
                    mode.append(self.out.samples()[i * self.chain_length + index][j])
                    max = self.out.samples()[i * self.chain_length + index][-1]

            # special case: only one chain
            if self.single_chain is not None:
                i = int(self.single_chain)
            print("Mode of chain %d with log posterior = %.7f is at:" % (i, max))

            # print in a format friendly for eos-scan-mc
            w = sys.stdout.write

            # all on one line
            w('"{ ')
            for p in mode:
                w("%+.5f " % p)
            w('}"\n')

            # 5 parameters per line
            """
            w("{ \n")
            for j, p in enumerate(mode):
                w("%+.5f " % p)
                if (j + 1) % 5 == 0:
                    w("\n")
            w("\n}\n")
            """
            sys.stdout.flush()

        # global mode
        self.global_mode_index = np.argmax(self._modes.T[-1])
        if self.single_chain is None:
            print("Global mode found in chain %d" % self.global_mode_index)

class PMC_Output(SamplingOutput):
    def _read(self, *args, **kwargs):
        # todo check automatically
        self.crop_outliers = kwargs.get('crop_outliers', 0)
        self.equal_weights = kwargs.get('equal_weights', False)
        self.hc_comp = kwargs.get('hc_comp', None)
        self.queue_output = kwargs.get('queue_output', None)
        self.step = kwargs.get('step', None)

        hdf5_file = h5py.File(self.input_file_name, 'r')

        # determine file type
        if self.queue_output is None:
            try:
                hdf5_file["/data/samples"]
                queue_output = True
            except:
                queue_output = False
        else:
            queue_output = self.queue_output

        # read samples
        step = 'final'
        if self.step is not None:
            step = str(self.step)

        try:
            if queue_output:
                samples = hdf5_file["/data/samples"][self.select[0]:self.select[1]]
            else:
                samples = hdf5_file["/data/" + step + "/samples"][self.select[0]:self.select[1]]
        except KeyError:
            samples = None
            print("No samples found")

        self.crop_last_columns = 3

        # read par defs
        par_defs, priors = read_descriptions(hdf5_file,
                                             data_set='descriptions/parameters',
                                             npar=samples.shape[1] - self.crop_last_columns,
                                             samples=samples)

        if samples is not None:
            # compute weights exactly once
            posterior = None
            if self.equal_weights:
                size = len(samples.T[-1])
                if self.select[0] and self.select[1]:
                    size = self.select[1] - self.select[0]
                self.weights = np.ones(size)
            elif queue_output:
                self.weights = np.exp(hdf5_file['/data/weights'][self.select[0]:self.select[1]].T['weight'])
                posterior = hdf5_file['/data/weights'][self.select[0]:self.select[1]].T['posterior']
            else:
                self.weights = np.exp(samples.T[-1][self.select[0]:self.select[1]])
                posterior = samples.T[-2][self.select[0]:self.select[1]]

            # find mode
            if posterior is not None:
                i_max = np.argmax(posterior)
                print("Found maximum posterior = %g with weight %g at" % (posterior[i_max], self.weights[i_max]))
                print(samples[i_max][:-3])

            # plot only a single component
            if self.step is not None:
                self.weights[samples.T[-3] != float(self.step)] = 0.0
                print('\033[91m' + 'WARNING: plotting only component %s' % self.step + '\033[0m')

            # reset highest 'outliers'
            if self.crop_outliers > 0:
                weight_clone = np.array(self.weights)
                weight_clone.sort()
                # need additional if counting backwards
                cut_off = weight_clone[-self.crop_outliers - 1]
                filter = self.weights > cut_off
                """
                posterior_clone = np.array(posterior)
                posterior_clone.sort()
                cut_off = np.log(crop_outliers) + posterior[i_max]
                print(cut_off)
                filter = posterior < cut_off
                print("posterior of crops")
                print(posterior[filter][0:10])
                print(self.weights[filter][0:10])
                """

                if posterior is not None:
                    print("P_{min} = %g, P_{max} = %g" % (np.min(posterior[filter][0:10]), np.max(posterior[filter][0:10])))
                    print("mean posterior of all samples = %g " % np.mean(posterior))
                    print("mean posterior of filtered samples = %g " % np.mean(posterior[filter]))
                    print(self.weights[filter][0:10])

                self.weights[filter] = 0.0
                print('\033[91m' + 'WARNING: filtering highest %d points from components' % len(np.where(filter)[0]) + '\033[0m')

            print("Samples have a total shape of %s " % str(samples.shape))
            print("Non-zero weights: %d " % len(np.where(self.weights > 0.0)[0]))

        # read statistics
        stats = []

        if queue_output:
            try:
                records = hdf5_file['/data/statistics'][:]
                for record in records:
                    stats.append((record[0], record[1], record[2]))

            except KeyError:
                pass
        else:
            step = 0
            mask = None
            while(True):
                try:
                    record = hdf5_file['/data/' + str(step) + "/statistics"][:]
                    comp_rec = hdf5_file['/data/' + str(step) + "/components"][:]
                    mask = comp_rec.T['weight'] > 0.0
                    stats.append((record[0][0], record[0][1], record[0][2], len(np.where(mask)[0])))
                except:
                    break
                step += 1
            # add last step
            try:
                    record = hdf5_file['/data/final/statistics'][:]
                    comp_rec = hdf5_file["/data/final/components"][:]
                    mask = comp_rec.T['weight'] > 0.0
                    stats.append((record[0][0], record[0][1], record[0][2], len(np.where(mask)[0])))
            except:
                pass

        # convert info to usable format
        usable_stats = np.array(stats)

        # read components
        if self.hc_comp is not None:
            if self.hc_comp == 'short':
                data_set_name = '/hc/input-components'
            elif self.hc_comp == 'long':
                data_set_name = '/hc/initial-guess'
        elif queue_output:
                data_set_name = '/data/initial/components'
        else:
            if self.step:
                step = self.step
            else:
                step = 'final'
            data_set_name = '/data/' + str(step) + '/components'

        print("Reading components from %s" % data_set_name)

        try:
            components = hdf5_file[data_set_name][:]
        except KeyError:
            raise Exception("Incorrect file format: use --pmc-queue-output ?")

        try:
            chol = hdf5_file[data_set_name].attrs['chol']
            if chol:
                print("Covariance available only in Cholesky decomposition. Converting it back.")
                n_dim = len(components.T['mean'][0])
                for i, mat in enumerate(components.T['covariance']):
                    cov = np.array(mat).reshape(n_dim, n_dim)

                    # filter upper/lower triangular matrix
                    l = np.tril(cov)
                    u = np.triu(cov)

                    # reassign flattened array
                    components.T['covariance'][i] = np.dot(l, u).ravel()

        except KeyError:
                pass

        print("Live components: %d out of %d" % (len(np.where(components.T['weight'] > 0)[0]), len(components)))

        # assign members
        self.samples = samples
        self.par_defs = par_defs
        self.priors = priors
        self.stats = usable_stats
        self.components = components

    def integrate(self, cuts=None):
        rows = self.filter(cuts)
        N = len(np.where(self.weights[rows])[0])
        print("Remaining samples in selected region: %d" % N)

        # sum partial weight
        partial_weight = np.sum(self.weights[rows])
        total_weight = np.sum(self.weights)
        ratio = partial_weight / total_weight

        # integral = average weight
        integral =partial_weight / len(self.samples)

        # normalize weights to avoid overflow if weights are large
        normalized_weights = self.weights[rows] / partial_weight
        error = partial_weight * np.sqrt(np.var(normalized_weights, ddof=1) / N)
        print('Partial weight: %g, total weight: %g, ratio: %g' % (partial_weight, total_weight, ratio))

        print('Integral of the selected region is: %g +- %g' % (integral, error))

        return (integral, ratio, total_weight, error)

    def component_integrate(self, cuts=None):
        if cuts:
            raise NotImplementedError("cuts not used!")
        # find nonzero component weights
        comp = np.where(self.components.T['weight'] > 0.0)[0]
        p = self.components.T['weight'][comp]
        N_total = len(self.samples)

        # initialization
        K = len(comp)
        Z = np.zeros((K,))
        var_Z = np.zeros((K,))
        inverse_var = 0.0

        # formulae and symbols in labbook, 28.02.2013
        # j is the component, and j_i its index
        # j_i != j if components are dead
        for j_i, j in enumerate(comp):
            # filter samples from each component
            # component index in column npar
            comp_indices = np.where(self.samples.T[len(self.par_defs)] == float(j))[0]
            weights = self.weights[comp_indices]

            # weight mean and aver
            E_w = np.mean(weights)
            var_w = np.var(weights, ddof=1)

            Z[j_i] = E_w

            # binomial variance
            N = len(weights)
            E_N = N_total * p[j_i]
            var_N = E_N * (1 - p[j_i])

            # variance of evidence
            var_Z[j_i] = E_N * var_w + E_w**2 * var_N
            var_Z[j_i] /= N**2

            #
            inverse_var += 1.0 / var_Z[j_i]

        # assume each Z[i] independent and normally distributed, but with diff. variance
        weighted_average = np.sum(Z / var_Z) / inverse_var

        rough_average = np.mean(Z)
        print("Estimate of Z with rough %g and Gauss %g" % (rough_average, weighted_average))

        # rough sample variance vs Gaussian average variance
        weighted_std_dev = np.sqrt(1.0 / inverse_var)
        rough_std_dev = np.sqrt(np.var(Z, ddof=1))

        # sample mean is an estimator with variance sigma^/N
        plain_std_dev = np.sqrt(np.var(self.weights, ddof=1) / len(self.weights))

        print("Std. deviation: rough sample %g vs Gauss weighted average %g vs plain weight variance %g" %
               (rough_std_dev, weighted_std_dev, plain_std_dev))
        print("Relative uncertainty estimate rough: %g" % (rough_std_dev / rough_average))
        print("Relative uncertainty estimate Gauss: %g" % (weighted_std_dev / weighted_average))
        print("One sigma interval Gauss: %s" %
               np.array(( weighted_average - weighted_std_dev, weighted_average + weighted_std_dev)))

        return (weighted_average, weighted_std_dev), (rough_average, rough_std_dev)

class MultinestOutput(SamplingOutput):

    def integrate(self, cuts=None):
        rows = self.filter(cuts)
        N = len(np.where(self.weights[rows])[0])
        print("Remaining samples in selected region: %d" % N)

        # sum partial weight
        partial_weight = np.sum(self.weights[rows])
        total_weight = np.sum(self.weights)
        ratio = partial_weight / total_weight

        print('Ratio: %g' % partial_weight)
        # evidence precomputed by multinest
        integral = ratio * np.exp(self.evidence)
        # symmetric error becomes asymmetric after exp
        # return larger of the two
        error = np.sqrt(np.max(np.exp(ratio * (self.evidence + self.evidence_error)) - integral,
                        integral - np.exp(ratio * (self.evidence - self.evidence_error))))

        print('Integral of the selected region is: %g +- %g' % (integral, error))

        return (integral, ratio, total_weight, error)

class EmceeOutput(SamplingOutput):
    def _read(self, *args, **kwargs):
        # read arguments
        self.chains = kwargs.get('chains', None)
        self.skip_initial = kwargs.get('skip_initial', 0)

        # parse basic info
        f = h5py.File(self.input_file_name, 'r')
        samples = f['/samples'][:]
        full_length = samples.shape[0]
        self.nwalkers = f.attrs['nwalkers']


        self.par_defs = []
        self.priors = []
        for i in xrange(samples.shape[1]):
            pd = ParameterDefinition(f['/descriptions/names'][i], f['/descriptions/min'][i],
                                     f['/descriptions/max'][i], False, False, False)

            self.par_defs.append(pd)
            self.priors.append(priorDistributions.Flat((pd.min, pd.max)))

        # validate arguments
        if self.select[0] is None:
            self.select[0] = 0
        if self.select[1] is None:
            self.select[1] = len(samples)
        if self.chains is not None:
            for c in self.chains:
                # positive indices in [0:nwalkers[
                # negative indices ok, but shift by one, valid range [-1:-nwalkers]
                if c >= self.nwalkers or -c > self.nwalkers:
                    raise KeyError('Selected chain %d exceed total number of chains (walkers) of %d' % (c, self.nwalkers))
        self.single_chain = None
        if hasattr(self.chains, '__len__') and len(self.chains) == 1:
            self.single_chain = str(self.chains[0])

        # adjust which range is drawn, default: full range
        if self.skip_initial is not None:
            # how many steps for each walker?
            nsteps = full_length / self.nwalkers
            print("Found %d steps per walker" % nsteps)
            # skip a fraction of steps from each walker to remove burn in
            self.select[0] = int(self.skip_initial * nsteps * self.nwalkers)

        # unselect all by default
        mask = np.zeros(len(samples), dtype=np.bool8)
        if self.chains is not None:
            # add a chain only if explicitly mentioned
            for c in self.chains:
                indices = range(self.select[0] + c, self.select[1], self.nwalkers)

                # assume all walkers stored together
                # so a single chain's samples scattered every nwalkers iterations
                mask[indices] = True
        else:
            # select all chains by default
            mask[self.select[0]:self.select[1]] = True

            # filter chains that have points in invalid region
            invalid_walkers = []
            stuck_walkers = []
            for l in xrange(self.nwalkers):
                indices = range(self.select[0] + l, self.select[1], self.nwalkers)
                for i, p in enumerate(self.par_defs):
                    skip = False
                    if samples[indices[0], i] == samples[indices[-1], i]:
                        skip = True
                        stuck_walkers.append(l)
                    a = np.ma.masked_outside(samples[indices, i], p.min, p.max, copy=False)
                    if a.mask.any():
                        skip = True
                        invalid_walkers.append((l, p.name))
                    if skip:
                        mask[indices] = False
                        break
            if invalid_walkers:
                print("Skipped %d walkers that contained values outside of the valid range (walker, parameter):"
                       % len(invalid_walkers))
                print(invalid_walkers)
            if stuck_walkers:
                print("Skipped %d walkers that seemed stuck in selected region:" % len(stuck_walkers))
                print(stuck_walkers)

        self.samples = samples[mask]

        # every selected sample has identical weight
        self.weights = np.ones(len(self.samples), dtype=np.float64)

        self.additional_info()

    def additional_info(self):
        """
        Print additional information about the samples

        """

        print("Total number of samples: %d from %d walkers" % (len(self.samples), self.nwalkers))
        try:
            import acor
        except ImportError:
            pass
        else:
            print("computing integrated autocorrelation time, mean, standard deviation:")
            try:
                tau = np.zeros(len(self.par_defs))
                for i in xrange(len(self.par_defs)):
                    results = acor.acor(self.samples[:, i])
                    tau[i] = results[0]
                    print("%s: %g, %g, %g" % (self.par_defs[i].name, tau[i],
                                                 results[1],  np.sqrt(np.var(self.samples.T[i], ddof=1))))
                print('')
                print('min max autocorrelation time: %g, %g' % (np.min(tau), np.max(tau)))
            except RuntimeError:
                    print("Skipping, probably because acor reports that " +
                          "The autocorrelation time is too long relative to the variance in dimension 1.")


#         bad_ind = np.where(self.samples.T[4] > 0.2)[0]
#         print("bad element at %d" % bad_ind[-1])
#         print(self.samples[bad_ind[-1]])