#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import commands
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as P

from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['text.usetex'] = False #not bool(commands.getstatusoutput('which latex')[0]) #requires LaTex installation
matplotlib.rcParams['text.latex.unicode'] = True

import numpy as np

import sys
import time

#get the figtree module, assume its directory is in the python path
try:
    import figtree
except:
    print("plotScript: Could not import figtree module")

import priors as priorDistributions

def histOutline(dataIn, *args, **kwargs):
    """
    Make a histogram that can be plotted with plot() so that
    the histogram just has the outline rather than bars as it
    usually does.

    Example Usage:
    binsIn = numpy.arange(0, 1, 0.1)
    angle = pylab.rand(50)

    ((bins, data), (raw_bins, raw_data)) = histOutline(binsIn, angle)
    plot(bins, data, 'k-', linewidth=2)

    """

    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return ((bins, data), (binsIn, histIn))

def find_hist_level(histo):
    """
    Find the **minimal** 68 % and 95% limit values from
    inverting the empirical CDF.
    Due to the discreteness of the PDF,
    the levels contain at least 68% [95%], thus overcover.
    Bin exceeding the alpha level are in the
    minimal credibility region containing alpha probability.

    Note:
    * procedure works in any number of dimensions
    * expect a histogram as an iterable with the actual, integer counts (or weights)
    """

    #turn 2D histo into flat histo if necessary
    #then sort
    bin_counts = np.sort(histo.ravel())

    #cumulative sum = cdf
    cdf = np.cumsum(bin_counts)

    #now start at back, the highest value, the full number of samples
    #minus -1 to overcover
    index = max(0, np.searchsorted(cdf, (1 - 0.68268949213708585) * cdf[-1]) - 1)
    level_68 = bin_counts[index]

    index = max(0, np.searchsorted(cdf, (1 - 0.95449973610364158) * cdf[-1]) - 1)
    level_95 = bin_counts[index]

    return (level_68, level_95)

def find_credibility_region_indices(histo, alpha):
    """
    histo = 1D histogram as iterable of bin counts

    Find max. bin and the smallest simply connected set of bins
    around that max. which contains at least alpha probability.

    Return list of indices of elements in the set.
    """

    #normalize histogram
    h = np.array(histo, dtype=float)
    h /= sum(h)

    max_index = h.argmax()
    max_count = h[max_index]
    indices = [max_index]

    a = max_index - 1 if max_index != 0 else None
    b = max_index + 1 if max_index != len(h) - 1 else None

    prob = max_count

    while prob < alpha:
        if a is None and b is None:
            raise Exception("Error: cannot find region")

        index = None
        if a is None:
            index = b
        elif b is None:
            index = a
        else:
            if h[a] > h[b]:
                index = a
            else:
                index = b

        if index == a:
            a = a - 1 if a > 0 else None
        else:
            b = b + 1 if b < len(h) - 1 else None

        indices.append(index)
        prob += h[index]

    return indices

#    print(max_count, max_index)
#    print(h)
#    print(h[indices])
#    print(sum(h[indices]))

def test_find_hist_region():
    histo = (1, 2, 3, 4, 5, 4, 2, 2, 1)
#    histo = (1, 2, 3, 4, 5, 4, 2, 2, 10)
#    histo = (12, 2, 3, 4, 5, 4, 2, 2, 10)
#    histo = (12, 1)
    print(find_credibility_region_indices(histo, alpha=0.68268))


def find_credibility_region(bin_edges, counts, alpha=0.68268):
    """
    Find the credibility region defined as the smallest simply
    connected region around the mode of a histogram containing
    at least alpha probability.

    Inputs:
    bin_edges = sequence of (N+1) values, all defined the left edge of a bin,
                except the last defines the right edge of the last bin.
    counts = sequence of N values, the counts in the bins
    alpha = the probability contained in the desired region

    Returns:
    (mode, sigma_lower, sigma_upper) = the distance from the maximum bin center to the left[right]-most
    bin center
    """

    indices = find_credibility_region_indices(counts, alpha)
    ind_sorted = np.sort(indices)

    #at bin center
    ind_max = indices[0]
    maximum = (bin_edges[ind_max] + bin_edges[ind_max + 1]) / 2.0
#    print("Edges of maximum bin = %s" % str(np.array([bin_edges[ind_max], bin_edges[ind_max + 1]])))

    #maximum at left edge
    if ind_sorted[-1] == 0:
        ind_lower = ind_max
        sigma_lower = 0.0
    else:
        ind_lower = ind_sorted[0]
        sigma_lower = maximum -(bin_edges[ind_lower] + bin_edges[ind_lower + 1]) / 2.0
    #maximum at right edge or no bins to right of maximum
    ind_upper = ind_sorted[-1]
    if ind_upper == len(bin_edges) - 1:
        sigma_upper = 0.0
    else:
        sigma_upper = (bin_edges[ind_upper] + bin_edges[ind_upper + 1]) / 2.0 - maximum

    return ((maximum, sigma_lower, sigma_upper), (ind_max, ind_lower, ind_upper))

def test_find_credibility_region():
    counts = (1, 2,2, 3,3, 4,4, 5, 4,3, 2, 1)
    bin_edges = np.linspace(1.1, 13, len(counts)+1 )

    print(counts)
    print(bin_edges)

    maximum, sigma_lower, sigma_upper = find_credibility_region(bin_edges, counts)
    print((maximum, sigma_lower, sigma_upper))


def filter_duplicates(array_in):
    """
    Filter out any duplicate values, assuming they are always next to each other
    counting in rows
    """

    #add column for multiplicity
    one_dimensional = False
    try:
        array_out = np.empty((array_in.shape[0], array_in.shape[1]+1))
    except IndexError:
        array_out = np.empty((array_in.shape[0], 2))
        one_dimensional = True
    #count unique entries
    index = 0

    #keep track of how often same event is seen
    counter = zeros((1,))

    for  row in range(array_in.shape[0]-1):
        counter += 1

        #unique entry
        if (array_in[ row+1] != array_in[ row]).any():
            array_out[index] = hstack( (array_in[row], counter) )
            index += 1
            counter = 0

    #add last element
    counter += 1
    array_out[index] = hstack( (array_in[-1], counter) )
    index += 1

    #crop result array
    if one_dimensional:
        array_out.resize((index, 2) )
    else:
        array_out.resize((index, array_in.shape[1]+1) )
    return array_out


def filter_duplicates_test():

    data = array( [[0,1], [0,1], [0,1], [0.3, 0.8], [0.3, 0.8], [3,1], [3,2], [3,2]] )

    filtered = filter_duplicates(data)
    print(filtered)

    assert sum(filtered.T[-1]) == len(data)

def find_runs(x, element):
    """
    Find all separated runs of one element in
    iterable x

    Example:

    x = (0, 1, 1, 1, 0, 0, 0, 1, 1)
    find_runs(x, 1) = [[1,4], [7,9]]
    """
    from copy import deepcopy

    runs = []

    #only runs of element
    in_run = (x[0] == element)
    run = [0, None] if in_run else [None, None]

    for i in range(1, len(x)):
        if x[i-1] != x[i]:
            if in_run:
                run[1] = i
                in_run = False
                runs.append(deepcopy(run))
            else:
                run[0] = i
                in_run = True

    # if run extends to the end of x
    if in_run:
        run[1] = len(x)
        runs.append(deepcopy(run))

    return runs

def test_runs():
    x = (0, 1, 1, 1, 0, 0, 0, 1, 1)
    print(find_runs(x, 1))
    assert find_runs(x, 1) == [[1,4], [7,9]]

    y = (1, 1, 0, 0)
    assert find_runs(y, 1) == [[0,2]]

    z = (1,1)
    assert find_runs(z, 1) == [[0,2]]

class PMC_Component(object):
    def __init__(self, weight, mean, variance, dof=None):
        self.weight = weight
        self.mean = mean
        self.variance = variance
        self.dof = dof

class DefaultTranslator(object):
    "Translator that keeps names unaltered."

    @staticmethod
    def to_tex(name):
        return name

class ParameterDefinition:

    def __init__(self, name, min, max, nuisance=False, discrete=False, index=None, n_major_ticks=None):
        self.name = name
        self.min = min
        self.max = max
        self.nuisance = nuisance
        self.discrete = discrete
        self.range = (min, max)
        self.i = index
        self.n_major_ticks = n_major_ticks

    def __repr__(self):
        return "{Name: %s, min: %g, max: %g, nuisance: %d, discrete: %d}" % \
                ( self.name, self.min, self.max, self.nuisance, self.discrete)

class MarginalDistributions:
    """
    DOCSTRING
    """

    def __init__(self, input_file_name, use_KDE=False, output_dir=None, input_source='pmc',
                 pmc_queue_output=False, pmc_equal_weights=False, pmc_crop_outliers=0,
                 chains=None, prerun=None, nuisance=True, select=(None, None), skip_initial=0,
                 projection=False, hc_comp=None):

        self.input_file_name = input_file_name

        #alternatively use KDE
        self.use_histogram = not use_KDE
        self.fixed_1D_binning = True
        self.one_dim_n_bins = 100
        self.fixed_2D_binning = True
        self.two_dim_n_bins = 100

        self.one_dimensional_only = False

        self.color_map = 'jet'
        self.use_contours = False

        if output_dir is None:
            self.__output_base_name = os.path.splitext(input_file_name)[0]
        else:
            self.__output_base_name = os.path.join(output_dir, os.path.split(os.path.splitext(input_file_name)[0])[1])

        #open a pdf file to hold the plots only when needed
        self.pdf_file = None

         #how many columns to spare (for log posterior or weights etc.)
        self.crop_last_columns = 1

        #population monte carlo
        self.input_source = input_source

        # plot proposal density instead of marginals, only useful with pmc
        self.proposal = False

        # repeat colors n times in proposal plots
        self.fold_color_spectrum = 0

        # nuisance
        self.use_nuisance = nuisance
        self.no_nuisance_vs_nuisance = False
        self.nuisance_2D = False

        ###
        #KDE settings
        ###

        #evaluate KDE at fewer points than for histogram
        self.kde_reduction = 1
        self.kde_interpolation = 'nearest' #'gaussian'
        # choose a value between 0 and 1, as a relative bandwidth
        self.kde_bandwidth = None
#        self.kde_boundary_pixel = 0.10 # 10 % of range

        self.kde_transform = False

        self.use_data_range = False
        self.scale_data_range = None

        #convert eos parameter names into nice latex names
        try:
            import translator
            self.tr = translator.EOS_Translator()
        except:
            print("plotScript: Could not find translator. Default to preserving names.")
            self.tr = DefaultTranslator()

        self.single_1D = None
        self.single_2D = None
        self.single_ext = ".pdf"

        # determine goodness of fit of this point
        # It is possible to any number of dimensions, just expect a dictionary like {0:3.2, 3:4.5},
        # where the key is the dimension index
        self.gof_point = {}

        self.plot_prior = True

        # store modes as determined from the chains or from HDF5 directly if available
        self.modes = []

        self.single_chain = None
        if hasattr(chains, '__len__') and len(chains) == 1:
            self.single_chain = str(chains[0])
        self.prerun = prerun

        ###
        # Integration options
        ###
        self.cuts = {}

        #parse the data
        self.select = select
        if self.input_source == 'pmc':
            self.data, self.par_defs, self.priors, self.stats, self.components = self.read_data_pmc(pmc_queue_output, pmc_equal_weights, pmc_crop_outliers, hc_comp)
            self.modes = None
            self.global_mode_index = None
        elif self.input_source == 'mcmc':
            self.data, self.par_defs, self.priors, self.modes = self.read_data_mcmc(chains, prerun, skip_initial)
            self.extract_chain_modes()
            self.stats = None
            self.components = None
        elif self.input_source == 'multinest':
            self.data, self.weights, self.par_defs, self.priors, self.evidence, self.evidence_error = self.read_data_multinest()
            self.modes = None
            self.global_mode_index = None
            self.stats = None
            self.components = None
        else:
            raise Exception("Unknown input source '%s'" % self.input_source)

        self.find_min_max()

        self.sigma = np.ones((self.data.shape[1]-self.crop_last_columns))
        self.nBins = np.empty((self.data.shape[1]-self.crop_last_columns),dtype=int)

        # Use as relative probability to max. All bins with prob
        # less than this will be whitenen in 2D plots
        self.minimum_probability = None

        ###
        # Marginalization options
        ###
        if projection:
            self.projection_labels = self.find_projection()

    def find_projection(self):
        """
        Determine credibility region in full parameter space, then project down.
        Use only the weights.
        """

        levels = find_hist_level(self.weights)

        labels = np.array(len(self.weights), dtype=Int)

        # no contour
        labels = 0

        # two sigma
        labels[np.where(self.weights > levels[1])] = 2

        # one sigma
        labels[np.where(self.weights > levels[0])] = 1

        return labels

    def read_data_txt(self):
        """
        read a text file consisting of columns.
        Initially for reading PMC output.
        Store weights in last column, where posterior is stored for MCMC
        """

        data = np.loadtxt(self.input_file_name)

        #define empty par defs
        par_defs = []

        for i in range(2, data.shape[1]):

            par_defs.append(ParameterDefinition("par_%d" % (i - 2),
                                np.min(data.T[i]), np.max(data.T[i]),
                                False, False))

        #reorder parameter columns so the coordinates appear first, and component/weight appear last
        shuffled_data = np.empty(data.shape)
        shuffled_data.T[-2:] = data.T[0:2]
        shuffled_data.T[:-2] = data.T[2:]

        #need to ignore two last columns
        self.crop_last_columns = 2

        #compute weights exactly once
        self.weights = np.exp(data.T[0])

        return shuffled_data, par_defs

    def read_data_multinest(self):
        """
        Parse multinest data using the pymultinest interface or from HDF5
        """

        par_defs = []
        priors = []
        if 'hdf5' in self.input_file_name:
            import h5py

            hdf5_file = h5py.File(self.input_file_name, 'r')

            try:
                samples = hdf5_file["/data/samples"][self.select[0]:self.select[1]]
            except KeyError:
                samples = None
                print("No samples found")


            if samples is not None:
                weights = hdf5_file["/data/weights"][self.select[0]:self.select[1]]

            evidence = hdf5_file["/data/weights"].attrs['log(evidence)']
            evidence_error = hdf5_file["/data/weights"].attrs['log(evidence) error']

        elif '.root' in self.input_file_name:
            import root_numpy as R

            tree_name = 'MultinestSamples'

            # copy parameters into a structured array
            struct_array = R.root2array(self.input_file_name, tree_name)

            # parameter name = branch name
            branches = R.list_branches(self.input_file_name, tree_name)

            # but weight is not a parameter name
            weights = struct_array['weight']
            weights = np.ones_like(weights)
            del branches[branches.index('weight')]

            # convert to a regular array (see http://wiki.scipy.org/Cookbook/Recarray)
            samples = struct_array[branches].view(float).reshape(len(struct_array), -1)

            print(np.sum(samples))

            # store at least parameter names, though ranges may be inaccessible for now
            for name in branches:
                par_defs.append(ParameterDefinition(name, np.min(struct_array[name]), np.max(struct_array[name]), False, False, False))
                priors.append(None)

            evidence = 0
            evidence_error = 0

        else:
            from pymultinest import Analyzer

            # todo Do I need ? Should be parsed automatically
            a = Analyzer(n_params=2, outputfiles_basename=self.input_file_name)

            # remove first two columns, so nothing superfluous
            samples = a.get_data().T[2:].transpose()

            weights = np.ascontiguousarray(a.get_data().T[0])

            # evidence
            evidence = a.get_stats()["global evidence"]
            evidence_error = a.get_stats()["global evidence error"]

        self.crop_last_columns = 0
        print("Parsed %d samples from multinest" % len(samples))
        print("resulting in log(Z) = %g +- %g" % (evidence, evidence_error))

        # mock up parameters definitions, no priors
        if not par_defs:
            for i,x in enumerate(samples.transpose()):
                par_defs.append(ParameterDefinition('par%d' % i, np.min(x), np.max(x), False, False, False))
                priors.append(None)

        return samples, weights, par_defs, priors, evidence, evidence_error

    def read_data_pmc(self, queue_output, equal_weights, crop_outliers, hc_comp):

        import h5py

        hdf5_file = h5py.File(self.input_file_name, 'r')

        # read samples
        step = 'final'
        if self.single_chain is not None:
            step = str(self.single_chain)

        try:
            if queue_output:
                samples = hdf5_file["/data/samples"][self.select[0]:self.select[1]]
            else:
                samples = hdf5_file["/data/" + step + "/samples"][self.select[0]:self.select[1]]
        except KeyError:
            samples = None
            print("No samples found")

        # read par defs
        par_defs = []
        priors = []
        f = priorDistributions.PriorFactory()
        descriptions = hdf5_file['descriptions/parameters'][:]
        for row in descriptions:
            par_defs.append(ParameterDefinition(row[0], row[1], row[2], row[3], False))
            print(row[4])
            try:
                prior_name, prior = f.create(row[4])
                assert(prior_name == row[0])
            except KeyError as e:
                prior = None
                print('Warning: in constructing prior for %s: %s' % (row[0], e.message))
            priors.append(prior)

        self.crop_last_columns = 3

        if samples is not None:
            #compute weights exactly once
            posterior = None
            if equal_weights:
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
            if self.single_chain is not None:
                self.weights[samples.T[-3] != float(self.single_chain)] = 0.0
                print('\033[91m' + 'WARNING: plotting only component %s' % self.single_chain + '\033[0m')

            # reset highest 'outliers'
            if crop_outliers > 0:
                weight_clone = np.array(self.weights)
                weight_clone.sort()
                # need additional if counting backwards
                cut_off = weight_clone[-crop_outliers - 1]
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

            print("Samples have a total shape of %s " % str(samples.shape) )
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
        if hc_comp is not None:
            if hc_comp == 'short':
                data_set_name = '/hc/input-components'
            elif hc_comp == 'long':
                data_set_name = '/hc/initial-guess'
        elif queue_output:
                data_set_name = '/data/initial/components'
        else:
            if self.single_chain:
                step = self.single_chain
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
                    components.T['covariance'][i] = np.dot(l,u).ravel()

        except KeyError:
                pass

        print("Live components: %d out of %d" % (len(np.where(components.T['weight'] > 0)[0]), len(components)))

        return samples, par_defs, priors, usable_stats, components

    def read_data_mcmc(self, chains, prerun, skip_initial):
        """
        New HDF5 output format since Nov 2011.

        Arguments:
        chains -- a sequence of integers to select a subset of available chains. None selects all chains

        """

        import h5py

        hdf5_file = h5py.File(self.input_file_name, 'r')

        n_chains_parsed = 0

        # fixme prerun won't work
        prefix = 'main run'
        if prerun:
            prefix = "prerun"

        # select all chains
        if chains is None:
            chains = range(len(list(hdf5_file[prefix])))

        first_chain = str(chains[0])
        self.n_chains = len(chains)

        #read data
        full_length = len(hdf5_file[prefix + '/chain #' + first_chain + "/samples"])

        #adjust which range is drawn, default: full range
        if self.select[0] is  None and skip_initial is not None:
            self.select[0] = int(skip_initial * full_length)

        merged_chains = hdf5_file[prefix + '/chain #' + first_chain + "/samples"][self.select[0]:self.select[1]]
        n_chains_parsed += 1

        #save shape info
        self.chain_length = len(merged_chains)

        #read out parameter info
        par_defs = []
        priors = []
        f = priorDistributions.PriorFactory()
        descriptions = hdf5_file['descriptions/' + prefix + '/chain #' + first_chain + "/parameters"][:]
        for row in descriptions:
            par_defs.append(ParameterDefinition(row[0], row[1], row[2], row[3], False))
            try:
                prior_name, prior = f.create(row[4])
                assert(prior_name == row[0])
            except KeyError as e:
                prior = None
                print('Warning: in constructing prior for %s: %s' % (row[0], e.message))

            priors.append(prior)

        # read out mode from stats: always last row
        stats = hdf5_file[prefix + '/chain #' + first_chain + "/stats/mode"]
        modes = [stats[-1]] #todo use modes in plot

        # read all remaining chains
        for chain in chains[1:]:
            data = hdf5_file[prefix + '/chain #%d/samples' % chain][self.select[0]:self.select[1]]
            merged_chains = np.concatenate((merged_chains, data), axis=0)
            modes.append(hdf5_file[prefix + '/chain #%d/stats/mode' % chain][-1])
            n_chains_parsed += 1

        hdf5_file.close()
        self.weights = np.ones(len(merged_chains))

        return merged_chains, par_defs, priors, np.array(modes)

    def read_data_hdf5_old(self, chain, prerun):
        """
        Use either  h5py [deprecated] or pytables to open the HDF5 file
        and merge the individual chains into one numpy array of size
        (nParam+1)*iterations*nChains
        """

        try:
            import tables as pytables

            hdf5_file = pytables.openFile(self.input_file_name,"r")

            #determine number of chains
            self.n_chains = len(hdf5_file.listNodes("/data"))

            n_chains_parsed = 0

            if chain is not None:
                first_chain = str(chain)
            else:
                first_chain = "0"

            prefix = ''
            if prerun:
                prefix = "prerun "

            #skip first chain
            node =  hdf5_file.getNode('/data', prefix + "chain #" + first_chain)

            # read in entire array
            # could also read in only 10 rows of first column with node[0:10,0]
            mergedChains = node.read()
            n_chains_parsed += 1
            print("Parsed %d chains" % n_chains_parsed)

            #save shape info
            self.chain_length = mergedChains.shape[0]

            #read out parameter info
            par_defs = []
            counter = 0
            for par_index in range(node.shape[1] - self.crop_last_columns):
                base_string = "FIELD_"+ str(counter)+"_"
                try:
                    par_defs.append(ParameterDefinition(node.attrs[base_string+"NAME"],
                                                    node.attrs[base_string+"ATTR_min"][0],
                                                    node.attrs[base_string+"ATTR_max"][0],
                                                    node.attrs[base_string+"ATTR_nuisance"][0],
                                                    node.attrs[base_string+"ATTR_discrete"][0]))
                except KeyError as err:
                    print("Please check attributes of data sets using h5ls -vr file/group/data_set")
                    raise err

                print(par_defs[-1])
                counter += 1

            par_defs.append(ParameterDefinition)
            self.weights = np.ones(len(mergedChains))

            #statistics
            for par_index in range(node.shape[1] - self.crop_last_columns):
                print("mean(%s) = %g" % (par_defs[par_index].name, np.mean(mergedChains.T[par_index][0:1000])))
                print("var(%s) = %g" % (par_defs[par_index].name, np.var(mergedChains.T[par_index][0:1000], ddof=1)))


            # plots based only on a single chain
            if chain is not None:
                print("Merge (pytables) has a total shape of %s " % str(mergedChains.shape) )
                return mergedChains, par_defs

            for chain in hdf5_file.listNodes("/data")[1:]:
                if chain.name[0:len(prefix) + 7] != prefix + 'chain #':
                    print("Skipping %s in group %s" % (chain.name, "/data"))
                    continue
                if chain.name == prefix + "chain #" + first_chain:
                    continue

                data = chain.read()

                #statistics
                print("chain %d" % (n_chains_parsed + 1) )
                for par_index in range(node.shape[1] - self.crop_last_columns):
                    print("mean(%s) = %g" % (par_defs[par_index].name, np.mean(data.T[par_index][0:1000])))
                    print("var(%s) = %g" % (par_defs[par_index].name, np.var(data.T[par_index][0:1000], ddof=1)))

                mergedChains = np.concatenate((mergedChains, data), axis=0)
                n_chains_parsed += 1

            print("Merge (pytables) has a total shape of %s " % str(mergedChains.shape) )
            print("Parsed %d chains" % n_chains_parsed)

            hdf5_file.close()

            #trivial weight one for all elements
            self.weights = np.ones(len(mergedChains))

            return mergedChains, par_defs


        except ImportError:
            print("Neither pytables nor h5py installed")


    def find_min_max(self):
        """
        Find minimum and maximum values of each parameter and mode of posterior.
        """
        if self.data is None:
            return
        self.min = np.empty((self.data.shape[1]-self.crop_last_columns,))
        self.max = np.empty((self.data.shape[1]-self.crop_last_columns,))

        for index in range(self.min.shape[0]):
            self.min[index] = min(self.data.T[index])
            self.max[index] = max(self.data.T[index])

    def find_limits_1D(self, index, method='ECDF'):
        """
        Find the central limits using the empirical CDF
        in 1D for the parameter identified by *index*
        """

        if method=='ECDF':
            #ascending order

            #find multiplicity of multiple events
            #it's in last column
            order_statistics = filter_duplicates( np.sort(self.data.T[index]) )

            #build cdf. normalize to unity
            ecdf = np.cumsum(order_statistics.T[-1])
            ecdf /= float(ecdf[-1])


            #binary search for index next to CDF value,
            #then extract physical parameter
            #subtract/add one from index to overcover
            limit_one_sigma = [order_statistics[max(0,np.searchsorted(ecdf, 0.15865525393145705)-1),0],
                               order_statistics[min(ecdf.shape[0]-1,np.searchsorted(ecdf, 0.84134474606854293)+1),0]]
            limit_two_sigma = [order_statistics[max(0,np.searchsorted(ecdf, 0.02275013194817919)-1),0],
                               order_statistics[min(ecdf.shape[0]-1,np.searchsorted(ecdf, 0.97724986805182079)+1),0]]
            return (limit_one_sigma, limit_two_sigma)

        if method == 'histogram':
            #bin the data
            histo, edges = np.histogram(self.data.T[index], self.nBins[index])

            #build CDF
            temp = np.cumsum(histo)
            cdf = temp / float(temp[-1])

            #binary search for index next to CDF value,
            #then extract physical parameter
            #subtract/add one from index to overcover
            limit_one_sigma = [edges[max(0,np.searchsorted(cdf, 0.15865525393145705)-1)],
                               edges[min(cdf.shape[0]-1, np.searchsorted(cdf, 0.84134474606854293)+2)]]

            limit_two_sigma = [edges[max(0,np.searchsorted(cdf, 0.02275013194817919)-1)],
                               edges[min(cdf.shape[0]-1, np.searchsorted(cdf, 0.97724986805182079)+2)]]
            return (limit_one_sigma, limit_two_sigma)

    @staticmethod
    def find_hist_limits(histo, credibilities=(0.68268949213708585, 0.95449973610364158), density=None):
        """
        Find the **minimal** 68 % and 95% level values from
        inverting the empirical CDF.
        Due to the discreteness of the PDF,
        the levels contain at least 68% [95%],
        thus overcovers.
        Note:
        * procedure works for one and 2D
        * expect a histogram with the actual, integer counts

        If **density**, check at what credibility level
         the given posterior density lies, i.e. starting from a mode,
         when lowering the water level, how much probability sticks out of
         the water when it equals the given level. Due to discreteness, we include
         the probability of the bin itself, thus we overcover. Should be negligible
         when many bins contribute, but makes a big difference if only a handful have
         nonzero probability. Therefore, we return (prob_overcovering, prob_no_overcovering)
          Do not return 68%, 95% levels.
        """

        #turn 2D histo into flat histo if necessary
        #then sort
        bin_counts = np.sort(histo.ravel())

        #cumulative sum = cdf
        cdf = np.cumsum(bin_counts)

        if density is not None:
            # where was level sorted to?
            # max to avoid negative index, -1 to overcover
            index = max(0, np.searchsorted(bin_counts, density) - 1)
            # how much is still above? In/exclude the bin with the point
            credibility_level_include = (cdf[-1] - cdf[index]) / cdf[-1]
            credibility_level_above =  (cdf[-1] - cdf[min(len(bin_counts) - 1, index + 1)]) / cdf[-1]
            return (credibility_level_include, credibility_level_above)

        levels = np.array(credibilities)
        for i,p in enumerate(credibilities):
            #now start at back, the highest value, the full number of samples
            #minus -1 to overcover
            index = max(0, np.searchsorted(cdf, (1 - p) * cdf[-1]) - 1)
            levels[i] = bin_counts[index]

        return levels

    def extract_chain_modes(self):
        """
        Find the mode of each chain and display it, ignoring the nuisance parameters.

        Assumes that each chain of same length.
        """
        print("There are %d chains, with a merged shape of %s" % (self.n_chains, self.data.shape))
        for i in range(self.n_chains):
            mode = []
            if len(self.modes) > 0:
                max = self.modes[i][-1]
                for j in range(self.data.shape[1] - self.crop_last_columns):
                    mode.append(self.modes[i][j])
            else:
                index = np.argmax(self.data.T[-1][i * self.chain_length : (i + 1) * self.chain_length])
                max = None
                for j in range(self.data.shape[1] - self.crop_last_columns):
                    if self.par_defs[j].nuisance and not self.use_nuisance:
                        continue
                    mode.append(self.data[i * self.chain_length + index][j])
                    max = self.data[i * self.chain_length + index][-1]

            #special case: only one chain
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

        #global mode
        self.global_mode_index = np.argmax(self.modes.T[-1])
        if self.single_chain is None:
            print("Global mode found in chain %d" % self.global_mode_index)

    def __bandwidth(self, samples, index):
        """
        Find the optimal bandwidth, assuming Gaussian distribution
        by using
        Scott's rule from
        A, D.W.S. & B, S.R.S. Multi-dimensional Density Estimation, p. 9 .
        at <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.85.7837>
        """
        self.sigma[index] = np.sqrt(np.var(samples, ddof=1))
        bandwidth = 5 * self.sigma[index] * np.power( samples.shape[0], -1/3.)
        self.nBins[index] = int(( self.max[index] - self.min[index])/bandwidth)
        return bandwidth

    def contours_one(self, x, densities, level, specifier, mode, **kwargs):
        """
        Plot contours and output intervals for one dim. distributions of parameter x.
        args:
        x = the ordinate
        densities = the estimate of the probability density
        level = the desired credibility level
        specifier = label used for text output, e.g. 'one sigma'
        mode = if hist, use traditional plotting.
        **kwargs are passed to the plotting routine of pylab.fill_between
        """
        if mode == 'hist':
            #draw 95% bins in yellow, 68% in green
            for i in range(len(x)-1):
                if densities[i] > level:
                    P.fill_between( (x[i], x[i+1]), 0, densities[i], **kwargs)
                    continue

            return

        high_points = densities >= level

        # find connected regions
        runs = find_runs(high_points, True)

        #green band for each interval
        for run in runs:
            P.fill_between(x[run[0]:run[1]], 0, densities[run[0]:run[1]], **kwargs)

        # print intervals and local modes
        intervals = []
        local_modes = []
        for run in runs:
            intervals.append([x[run[0]], x[run[1] - 1]])
            intervals[-1].append(intervals[-1][1] - intervals[-1][0])
            local_modes.append(x[run[0] + np.argmax(densities[run[0]:run[1]])])

        print("Minimal %s interval(s):" % specifier)
        print(np.array(intervals))

        print("Local marginalized mode(s) at:")
        print(np.array(local_modes))

    def contours_two(self, x_range, y_range, densities, desired_levels=None, color='blue', grid=True, line=False):
        """
        Plot several filled 2D contours
        args:
        x = the pixel centers in parameter values (NxN array)
        densities = probability density estimate at x (NxN array)
        levels = the probability levels
        """

        # pass pixel values as histogram:
        # interpret value at bin center as average over bin
        # since all bins have same size, the integral in bin
        # is just mean value times volume, and the volume cancels

        levels = self.find_hist_limits(densities).flatten().tolist()

        # default: use 1 and 2 sigma
        if desired_levels == '1 sigma':
            levels.remove(levels[1])
        elif desired_levels == '2 sigma':
            levels.remove(levels[0])

        # add extra levels for min and max to fill contours correctly
        levels.append(0)
        levels.reverse()
        levels.append(np.max(densities))

        # draw filled areas and the contour
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

        artist = None
        if line:
            artist = P.contour(densities, levels[1:-1], colors=color, extent=extent)
            P.setp(artist.collections[0], linestyle='dashed')
        else:
            artist = P.contourf(densities,
                            levels,
                            colors=['white', color, color],
                            extent=extent)

            # make 'outer' contour fill transparent, and inner opaque
            zc = artist.collections[0]
            P.setp(zc, alpha=1)

            zc = artist.collections[1]
            P.setp(zc, alpha=0.4)

        # add grid lines for easier visual comparison
        if grid:
            P.grid()
        ax = P.gca()
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        return artist

        # colorbar uses info from contours
        # P.colorbar(artist)

        # output "corners" of the contours
        """
        maxima_68 = []
        maxima_95 = []
        minima_68 = []
        minima_95 = []
        for c in artist.collections[1]._paths:
            minima_95.append((np.min(c.vertices.T[0]), np.min(c.vertices.T[1])))
            maxima_95.append((np.max(c.vertices.T[0]), np.max(c.vertices.T[1])))
        try:
            for c in artist.collections[2]._paths:
                minima_68.append((np.min(c.vertices.T[0]), np.min(c.vertices.T[1])))
                maxima_68.append((np.max(c.vertices.T[0]), np.max(c.vertices.T[1])))
        except IndexError:
            pass

        print("Minima of 95 %% contours at = %s " % np.array(minima_95))
        print("Maxima of 95 %% contours at = %s " % np.array(maxima_95))
        print("Minima of 68 %% contours at = %s " % np.array(minima_68))
        print("Maxima of 68 %% contours at = %s " % np.array(maxima_68))
        """

    def __title(self):

        title = ''

        if self.single_chain:
            if self.input_source == 'pmc':
                title = " pmc\;step\; \#" + self.single_chain
            elif self.input_source == 'mcmc':
                title = "chain\;\#" + self.single_chain
            else:
                title = ''

            if matplotlib.rcParams['text.usetex']:
                title = r"${\mathrm " + title + r"}$"
        else:
            if self.input_source == 'mcmc':
                title = r"${\mathrm{All}\; \mathrm{chains}}$"
            else:
                pass

        return title

    def __extract_smallest_intervals(self, x, prob_density, level_68, level_95):
        """
        Extract set of smallest intervals in 1D
        """
        points_95 = prob_density >= level_95

        # find connected regions
        runs = find_runs(points_95, True)

        # print intervals
        intervals_95 = []
        local_modes = []
        for run in runs:
            print(run)
            intervals_95.append([x[run[0]], x[run[1] - 1]])
            intervals_95[-1].append(intervals_95[-1][1] - intervals_95[-1][0])
            mode_index = run[0] + np.argmax(prob_density[run[0]:run[1]])
            local_modes.append((x[mode_index] + x[mode_index + 1]) / 2.0)
        print("Minimal 2 sigma intervals:")
        print(np.array(intervals_95))
        print("Local marginalized mode(s):")
        print(np.array(local_modes))

        ###
        # one sigma intervals
        ###
        points_68 = prob_density >= level_68

        # find connected regions
        runs = find_runs(points_68, True)

        # print intervals
        intervals_68 = []
        for run in runs:
            intervals_68.append([x[run[0]], x[run[1] - 1]])
            intervals_68[-1].append(intervals_68[-1][1] - intervals_68[-1][0])

        print("Minimal 1 sigma intervals:")
        print(np.array(intervals_68))

    def one_dimensional(self, index,
                        prior_style=dict(color='black', linestyle='dashed'),
                        marginal_style=dict(color='black', linestyle='solid', linewidth=0.5),
                        minor_locator=matplotlib.ticker.AutoMinorLocator(),
                        legend_label='', prior_label=''):
        """
        Expect vector of length 1xN.
        Parameter identified by index (0..n)
        """


        #do nothing if one of the parameters is a nuisance parameter
        if not self.use_nuisance and self.par_defs[index].nuisance:
            return False

        samples = self.data.T[index]

        #Scott's rule from
        #A, D.W.S. & B, S.R.S. Multi-dimensional Density Estimation, p. 9 .
        #at <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.85.7837>

        if  not self.use_histogram:
            bandwidth = self.__bandwidth(samples, index)
#            self.sigma[index] = np.sqrt(np.var(samples, ddof=1))
#            bandwidth = 5 * self.sigma[index] * np.power( samples.shape[0], -1/3.)
#            self.nBins[index] = int(( self.max[index] - self.min[index])/bandwidth)
        if self.fixed_1D_binning:
            self.nBins[index] = self.one_dim_n_bins
        if self.kde_bandwidth is not None:
            bandwidth = self.kde_bandwidth

        x_min = self.par_defs[index].min
        x_max = self.par_defs[index].max

        #older files didn't have parameter info
        if x_min == x_max:
            x_min = self.min[index]
            x_max = self.max[index]
        if self.use_data_range:
            x_range = self.max[index] - self.min[index]
            x_min = self.min[index] - self.scale_data_range * x_range / 2.0
            x_max = self.max[index] + self.scale_data_range * x_range / 2.0
        if index in self.cuts:
            x_min = self.cuts[index][0]
            x_max = self.cuts[index][1]

        # extract parameter name
        parameter_name = self.tr.to_tex(self.par_defs[index].name)
        if parameter_name == '':
            parameter_name = "par"+str(index)

        print("%s: x_range = [%g, %g], bins = %d" % (parameter_name, x_min, x_max, self.nBins[index]))

        #KDE doesn't make sense for discrete parameters
        if self.par_defs[index].discrete:
            print("found a discrete parameter")
            print(self.use_histogram)
            histogram_state = self.use_histogram
            self.use_histogram = True

        # general array, either from histogramming or from KDE
        probability_array = None

        if self.use_histogram:
             print("Using histogram")
             (hist_outline, hist_normal) = histOutline(samples, bins=self.nBins[index],  weights=self.weights,
                                                       normed=True, range=(x_min, x_max))
             P.plot(hist_outline[0], hist_outline[1], label=legend_label, **marginal_style)

             # the data used for credibility regions
             probability_array = hist_normal[1]

             if self.nBins[index] < 4:
                print(probability_array)

             if self.use_contours:

                 level_68, level_95 = self.find_hist_limits(hist_normal[1])

                 self.__extract_smallest_intervals(hist_normal[0], probability_array, level_68, level_95)

                 # draw 95% bins in light blue, 68% in opaque blue
                 self.contours_one(hist_normal[0], hist_normal[1], level_95, 'two_sigma', 'hist', facecolor='blue', alpha=0.4, linewidth=0, edgecolor='none')
                 self.contours_one(hist_normal[0], hist_normal[1], level_68, 'one_sigma', 'hist', facecolor='blue', alpha=1, linewidth=0, edgecolor='none')

                 """
                 for i in range(len(hist_normal[0])-1):

                    if hist_normal[1][i] > level_68:
                        P.fill_between( (hist_normal[0][i], hist_normal[0][i+1]), 0, hist_normal[1][i], facecolor='green', alpha=1, linewidth=0)
                        continue

                    if hist_normal[1][i] > level_95:
                        P.fill_between( (hist_normal[0][i], hist_normal[0][i+1]), 0, hist_normal[1][i], facecolor='yellow', alpha=1 ,edgecolor='none')
                        continue
                """
        else: #use KDE
            print("Using KDE")
            mesh_points = np.linspace(x_min, x_max, self.nBins[index] )

            #setup for figtree
            start_time = time.time()
            densities = figtree.figtree(np.ascontiguousarray(samples), mesh_points, weights=self.weights,  bandwidth=bandwidth, eval="auto")
            end_time = time.time()
            print("figtree used %f s" % (end_time-start_time) )

            from scipy import interpolate
            #do a spline interpolation on a finer grid
            tck = interpolate.splrep(mesh_points, densities, s=0)
            finer_mesh = np.linspace(x_min, x_max, 5*self.nBins[index] )
            densities_interp = interpolate.splev(finer_mesh, tck, der=0)

            # can't use interpolated array, because x_min/x_max for bin index finding won't work
            probability_array = densities

            # normalize to unity
            densities_interp /= np.sum(densities_interp * (x_max - x_min) / len(finer_mesh))

            P.plot(finer_mesh, densities_interp, label=legend_label, **marginal_style)

            # todo why is this almost a copy of _extract* ?
            #now add credibility bands
            if self.use_contours:
                ###
                #minimal interval
                ###
                level_68, level_95 = self.find_hist_limits(densities_interp)

                points_95 = densities_interp >= level_95

                # find connected regions
                runs = find_runs(points_95, True)

                #green band for each interval
                for run in runs:
                    P.fill_between(finer_mesh[run[0]:run[1]], 0, densities_interp[run[0]:run[1]], facecolor='yellow', alpha=0.6)

                # print intervals
                intervals_95 = []
                local_modes = []
                for run in runs:
                    intervals_95.append([finer_mesh[run[0]], finer_mesh[run[1] - 1]])
                    intervals_95[-1].append(intervals_95[-1][1] - intervals_95[-1][0])
                    local_modes.append(finer_mesh[run[0] + np.argmax(densities_interp[run[0]:run[1]])])
                print("Minimal 2 sigma intervals:")
                print(np.array(intervals_95))

                ###
                # one sigma intervals
                ###
                points_68 = densities_interp >= level_68

                # find connected regions
                runs = find_runs(points_68, True)

                #green band for each interval
                for run in runs:
                    P.fill_between(finer_mesh[run[0]:run[1]], 0, densities_interp[run[0]:run[1]], facecolor='green', alpha=1)

                # print intervals
                intervals_68 = []
                for run in runs:
                    intervals_68.append([finer_mesh[run[0]], finer_mesh[run[1] - 1]])
                    intervals_68[-1].append(intervals_68[-1][1] - intervals_68[-1][0])

                print("Minimal 1 sigma intervals:")
                print(np.array(intervals_68))
                print("Local marginalized mode(s):")
                print(np.array(local_modes))

        # determine goodness-of-fit
        if self.gof_point is not None:
            try:
                # read out value in this this dimension
                value = self.gof_point[index]
                # calculate index of bin containing the point
                bin_index = np.floor((value - x_min) / (x_max - x_min) * self.nBins[index])
                # special case: value at right edge
                bin_index = min(bin_index, self.nBins[index] - 1)

                posterior_level = probability_array[bin_index]

                (prob_greater_equal, prob_greater_than) = self.find_hist_limits(probability_array, density=posterior_level)
                sigmas_ge = Gaussian.ppf((prob_greater_equal + 1) / 2.0, location=0, scale=1)
                sigmas_gt = Gaussian.ppf((prob_greater_than + 1) / 2.0, location=0, scale=1)

                print("GoF: point (%g) at %g%% level (w/o overcovering: at %g%% level). On the Gaussian scale, that's at %g [%g] sigmas" %
                      (value, prob_greater_equal * 100, prob_greater_than * 100, sigmas_ge, sigmas_gt))

                P.plot([value, value], [0.0, posterior_level], linestyle='dashed', linewidth=0.8)

            except KeyError:
                pass

        if self.par_defs[index].discrete:
            self.use_histogram = histogram_state

        # plot global mode for single chains
        if self.global_mode_index is not None:
            P.scatter(self.modes[self.global_mode_index][index], 0, marker='^', s=200)

        # plot the prior in the same plot
        if self.plot_prior and self.priors[index] is not None:
            prior = self.priors[index]
            integral = 1
            mesh_points = np.linspace(x_min, x_max, self.nBins[index] )

            prior_values = prior.evaluate(mesh_points)

            P.plot(mesh_points, prior_values, label=prior_label, **prior_style)
        P.xlim( (x_min, x_max) )
        P.ylim(0)

        P.xlabel(parameter_name)
        if minor_locator:
            ax = P.gca()
            ax.xaxis.set_minor_locator(minor_locator)

        P.title(self.__title())

        return True

    def two_dimensional(self, par1, par2):

        #do nothing if one of the parameters is a nuisance parameter
        if (not self.use_nuisance or not self.nuisance_2D) and (self.par_defs[par1].nuisance or self.par_defs[par2].nuisance):
            return False

        #don't plot one nuisance vs another nuisance parameter
        if self.no_nuisance_vs_nuisance and (self.par_defs[par1].nuisance and self.par_defs[par2].nuisance):
            return False

        #2D histogram
        samples1 = self.data.T[par1]
        samples2 = self.data.T[par2]

        # determine bandwidths
        self.__bandwidth(samples1, par1)
        self.__bandwidth(samples2, par2)

        ###
        #plotting range
        ###
        x_min = self.par_defs[par1].min
        x_max = self.par_defs[par1].max
        y_min = self.par_defs[par2].min
        y_max = self.par_defs[par2].max

        #old files don't have range in HDF5
        if x_min == x_max:
            x_min = self.min[par1]
            x_max = self.max[par1]
        if y_min == y_max:
            y_min = self.min[par2]
            y_max = self.max[par2]
        if self.use_data_range:
            range = self.max[par1] - self.min[par1]
            x_min = self.min[par1] - self.scale_data_range * range / 2.0
            x_max = self.max[par1] + self.scale_data_range * range / 2.0
            range = self.max[par2] - self.min[par2]
            y_min = self.min[par2] - self.scale_data_range * range / 2.0
            y_max = self.max[par2] + self.scale_data_range * range / 2.0
        if par1 in self.cuts:
            x_min = self.cuts[par1][0]
            x_max = self.cuts[par1][1]
        if par2 in self.cuts:
            y_min = self.cuts[par2][0]
            y_max = self.cuts[par2][1]

        twoD_bins = np.array((self.nBins[par1], self.nBins[par2]), dtype=int)
        if self.fixed_2D_binning:
            twoD_bins = np.array((self.two_dim_n_bins, self.two_dim_n_bins), dtype=int)

        print("Grid shape %s for parameters %s" % (twoD_bins, [self.par_defs[par1].name, self.par_defs[par2].name]))

        #KDE doesn't make sense for discrete parameters
        if self.par_defs[par1].discrete or self.par_defs[par2].discrete:
            histogram_state = self.use_histogram
            self.use_histogram = True
            contour_state = self.use_contours
            self.use_contours = False

        # general 2D array of probablity (density) in pixels
        probability_array = None

        # prepare for whitening
        cmap = P.get_cmap(self.color_map)
        if self.minimum_probability is not None:
            cmap.set_under('white')

        if self.use_histogram:
            H, xedges, yedges = np.histogram2d(samples1, samples2 ,
                                            bins= (np.linspace(x_min, x_max , twoD_bins[0] + 1), np.linspace(y_min, y_max , twoD_bins[1] + 1) ),
                                            weights=self.weights)

            #convert to standard display order
            probability_array = np.fliplr(np.rot90(H,k=3))

            #Acceptable values are None, ‘nearest’, ‘bilinear’, ‘bicubic’, ‘spline16’, ‘spline36’, ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’, ‘quadric’, ‘catrom’, ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
            interpolation_method = 'nearest'

            # everything below will be whitened
            vmin = self.minimum_probability * np.max(probability_array) if self.minimum_probability is not None else 0.0

            #add contours
            if self.use_contours:
                self.contours_two((xedges[0], xedges[-1]), (yedges[0], yedges[-1]), probability_array)
            #plot colored density
            else:
                #imshow has opposite orientation
                P.imshow(probability_array,
                         cmap=cmap,
                         vmin=vmin,
                         extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                         origin='lower',
                         interpolation=interpolation_method)
        # KDE
        else:

            #join samples, 2 rows, N columns. One sample per column
            samples = np.c_[samples1, samples2].T

            #use less points on each axis
            twoD_bins = twoD_bins/ self.kde_reduction

            #boundary pixel width in true coordinates
            dx = (x_max - x_min)/float(twoD_bins[0])
            dy = (y_max - y_min)/float(twoD_bins[0])

            # create grid of points at which density is estimated
            X, Y = np.mgrid[x_min + dx/2.:x_max - dx/2.:complex(0, twoD_bins[0] ), \
                         y_min + dy/2.:y_max - dy/2.:complex(0, twoD_bins[1])]

            mesh_points = np.c_[X.ravel(), Y.ravel()].T

            mesh_points[0,:] = (mesh_points[0,:] - x_min)/(x_max - x_min)
            mesh_points[1,:] = (mesh_points[1,:] - y_min)/(y_max - y_min)

            #setup for figtree
            start_time = time.time()

            #use rule of thumb from Scott, Sain (1992) after Eq. 19
            if self.kde_bandwidth is None:
                 estimated_bandwidth = 1/4.0 * np.power(samples.shape[0], -1 / (2 + 4.0))
            else:
                estimated_bandwidth = self.kde_bandwidth

            #rescale coordinates to unit hypercube (fast)
            if not self.kde_transform:
                transformed_samples = np.empty( (len(samples1),2))
                transformed_samples[:,0] = (samples1 - x_min)/(x_max - x_min)
                transformed_samples[:,1] = (samples2 - y_min)/(y_max - y_min)

            #transform into "almost principal components"
            #following Scott, Sain (1992) 3.3
            else:
                covariance = np.cov(samples, rowvar=0)
                A = np.matrix(covariance / np.sqrt(np.linalg.det(covariance)))

                # A^{-1/2} in Scott, Sain
                import scipy.linalg
                B = scipy.linalg.matfuncs.sqrtm(scipy.linalg.inv(A))
                #convert to real from complex
                B = np.matrix(B, dtype=float)

                #transform the input data
                transformed_samples = np.array([np.ravel(B * np.matrix(sample).transpose()) for sample in samples] )
                print(transformed_samples[0])

                #transform the mesh point corners and create new mesh
                ac = B * np.matrix( [X[0,0], Y[0,0]]).transpose()
                bd = B * np.matrix( [X[-1,-1], Y[-1,-1]]).transpose()

                X, Y = np.mgrid[ac[0,0]:bd[0,0]:complex(0, twoD_bins[0] ), \
                     ac[1,0]:bd[1,0]:complex(0, twoD_bins[1])]
                mesh_points = np.c_[X.ravel(), Y.ravel()].T

            verbosity=False

            #do the work from the ctypes wrapper to the C-code
            estimated_pdf = figtree.figtree(transformed_samples,
                                            np.ascontiguousarray(mesh_points.T[:]),
                                            self.weights,
                                            bandwidth= estimated_bandwidth,
                                            verbose=verbosity)
            end_time = time.time()
            print("figtree used %f s" % (end_time-start_time) )

            #turn density from vector into matrix again
            probability_array = np.reshape(estimated_pdf.T, X.shape)

            # transform such that plot has usual orientation
            probability_array = np.fliplr(np.rot90(probability_array ,k=3))

            # everything below will be whitened
            vmin = self.minimum_probability * np.max(probability_array) if self.minimum_probability is not None else 0.0

            if self.use_contours:
                self.contours_two((x_min, x_max), (y_min, y_max), probability_array)
            else:
                #imshow has opposite orientation
                P.imshow(probability_array,
                        cmap=cmap,
                        vmin=vmin,
                        extent=[x_min, x_max, y_min, y_max],
                        origin='lower',
                        interpolation = self.kde_interpolation)

        # whiten part with extremely low probability
        """
        if self.minimum_probability is not None:
            CS = P.contourf(probability_array,
                    [self.minimum_probability * np.max(probability_array), np.max(probability_array)],
                    colors='white',
                    extend='both',
                    extent=[x_min, x_max, y_min, y_max])
            P.setp(CS.collections[1], alpha=0.0)
        """
        if self.par_defs[par1].discrete or self.par_defs[par2].discrete:
            self.use_histogram = histogram_state
            self.use_contours = contour_state

        # determine goodness-of-fit
        if self.gof_point is not None:
            try:
                # read out value in this this dimension
                value = (self.gof_point[par1], self.gof_point[par2])

                # calculate index of bin containing the point
                # special case: value at right edge
                # subtract one extra because we use one bin less. Why actually?
                bin_index = (min(int((value[0] - x_min) / (x_max - x_min) * twoD_bins[0]), twoD_bins[0] - 1),
                             min(int((value[1] - y_min) / (y_max - y_min) * twoD_bins[1]), twoD_bins[1] - 1))

                posterior_level = probability_array[bin_index]

                (prob_greater_equal, prob_greater_than) = self.find_hist_limits(probability_array, density=posterior_level)

                sigmas_ge = Gaussian.ppf((prob_greater_equal + 1) / 2.0, location=0, scale=1)
                sigmas_gt = Gaussian.ppf((prob_greater_than + 1) / 2.0, location=0, scale=1)
                print("GoF: point %s at %g%% level (w/o overcovering: at %g%% level). On the Gaussian scale, that's at %g [%g] sigmas" %
                      (str(value), prob_greater_equal * 100, prob_greater_than * 100, sigmas_ge, sigmas_gt))

                P.scatter(value[0], value[1], marker='+', color='k')
            except KeyError:
                pass

        #set labels, avoid empty parameter names
        x_label = self.tr.to_tex(self.par_defs[par1].name)
        y_label = self.tr.to_tex(self.par_defs[par2].name)

        if x_label =="":
            x_label = "par"+str(par1)
        if y_label == "":
            y_label = "par"+str(par2)

        P.xlabel(x_label)
        P.ylabel(y_label)

        P.title(self.__title())

        #without this call, length of axis on image corresponds to size of data.
        #thus if one param in [0.3, 0.34] and the other in [0, 6], the image is just one strip
        P.axis('tight')

        return probability_array

    def evolution(self, scale='log', posterior=False):
        """
        Plot evolution of single chains in 1D
        """

        title = 'All chains'
        if self.single_chain is not None:
            title = 'chain ' + str(self.single_chain)

        self.pdf_file = PdfPages(self.__output_base_name + "_evol.pdf")

        if posterior:
            max_posterior = max(self.data.T[-1])
            min_posterior = min(self.data.T[-1][0.05*len(self.data):])



        #print out mode info
        for par in range(self.data.shape[1]-self.crop_last_columns):

            #set labels, avoid empty parameter names
            y_label = self.tr.to_tex(self.par_defs[par].name)

            if y_label == "":
                y_label = "par"+str(par)

            fig = P.figure(figsize=(16,9))
            ax1 = fig.add_subplot(111)

            x_min = self.select[0] if self.select[0] is not None else 0
            x_max = self.select[1] if self.select[1] is not None else self.chain_length

            for n in range(self.n_chains):
                l1 = ax1.plot(np.arange(x_min, x_max),
                              self.data.T[par][n * self.chain_length:(n + 1) * self.chain_length], label=r"chain \#" + str(n))

            ax1.set_xscale(scale)
            ax1.set_xlim(x_min, x_max)


            ax1.set_xlabel("Iterations")
            ax1.set_ylabel(y_label)

            ###
            #plotting range
            ###
            y_min = self.par_defs[par].min
            y_max = self.par_defs[par].max

            #old files don't have range in HDF5
            if y_min == y_max:
                y_min = self.min[par]
                y_max = self.max[par]
            if self.use_data_range:
                y_range = self.max[par] - self.min[par]
                y_min = self.min[par] - self.scale_data_range * y_range / 2.0
                y_max = self.max[par] + self.scale_data_range * y_range / 2.0

            ax1.set_ylim(y_min, y_max)
            if self.single_chain is None:
                P.legend(loc='upper left')

            #rescale to coordinate range
            if posterior and (self.single_chain is not None):
                ax2 = ax1.twinx()
                l2 = ax2.plot(self.data.T[-1], label='$\log P$', color='black')
                ax2.set_ylabel('$\log P$')
                ax2.set_ylim(min_posterior, max_posterior)
                P.legend(loc='upper left')

            P.title(title)

            self.pdf_file.savefig()
            P.close()

    def convergence(self):
        """
        Plot evolution of convergence diagnostics
        """
        if not self.stats.any():
            print("No convergence statistitics available")
            return

        print("Stats:")
        print("perplexity\t\tESS\t\tevidence\t\tlive comp.")
        print(self.stats.__repr__())

        self.pdf_file = PdfPages(self.__output_base_name + "_stats.pdf")

        if True:
            P.figure(figsize=(8,8))
            P.plot(self.stats.T[0], label='perplexity')
            P.plot(self.stats.T[1], label='eff. sample size')
            P.ylim((0,1))
            P.legend(loc='upper left')
            P.title("convergence diagnostics")
            P.xlabel("steps")
            self.pdf_file.savefig()

            P.figure(figsize=(8,8))
            P.plot(self.stats.T[2], label='evidence')
            P.title('evidence')
            P.xlabel("steps")

            self.pdf_file.savefig()

            self.pdf_file.close()
            P.close()

    def comp_integrate(self):
        """Compute total evidence and estimate uncertainty from combination of individual components"""

        assert(self.input_source is 'pmc')

        # find nonzero component weights
        comp = np.where(self.components.T['weight'] > 0.0)[0]
        p = self.components.T['weight'][comp]
        N_total = len(self.data)

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
            comp_indices = np.where(self.data.T[len(self.par_defs)] == float(j))[0]
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

    def integrate(self):
        """
        Calculate the integral in a subregion.
        Doesn't work with MCMC, only with PMC or multinest.
        """

        assert(self.input_source is not 'mcmc')

        # samples inside of integration region
        rows = range(len(self.data))
        if self.cuts:
            # apply each cut
            print("Using the following cuts to define the integration region: " % self.cuts)
            for par, cut in self.cuts.iteritems():
                print("--par: %s, min: %g, max: %g" % (self.par_defs[par].name, cut[0], cut[1]))

             # first apply lower, then upper cuts
            rows1 = np.logical_and.reduce([self.data[:, par] >= cut[0] for par, cut in self.cuts.iteritems()])
            rows2 = np.logical_and.reduce([self.data[:, par] <= cut[1] for par, cut in self.cuts.iteritems()])
            rows = np.logical_and(rows1, rows2)

        N = len(where(self.weights[rows])[0])
        print("Remaining samples in selected region: %d" % N)

        # sum partial weight
        partial_weight = np.sum(self.weights[rows])
        total_weight = np.sum(self.weights)
        ratio = partial_weight / total_weight
        error = None

        if self.input_source == 'pmc':
            # integral = average weight
            integral = partial_weight / len(self.data)
            error = np.sqrt(np.var(self.weights[rows], ddof=1) / N)
            print('Partial weight: %g, total weight: %g, ratio: %g' % (partial_weight, total_weight, ratio))
        elif self.input_source == 'multinest':
            print('Ratio: %g' % partial_weight)
            # evidence precomputed by multinest
            integral = ratio * np.exp(self.evidence)
            # symmetric error becomes asymmetric after exp
            # return larger of the two
            error = np.sqrt(np.max( np.exp(ratio * (self.evidence + self.evidence_error)) - integral,
                            integral - np.exp(ratio * (self.evidence - self.evidence_error))))

        print('Integral of the selected region is: %g +- %g' % (integral, error))

        return (integral, ratio, total_weight, error)

    def proposal_2D(self, par1, par2, centers=False, solid_edge=False, **kwargs):
        """
        Plot the proposal distribution (PMC).
        """

        assert(self.input_source is 'pmc')

        from numpy import linalg
        from matplotlib.patches import Ellipse
        from matplotlib.cm import get_cmap

        mask = self.components.T['weight'] > 0.0

        # positions

        nCols =  len(self.par_defs)

        cmap = get_cmap(name='spectral')
        colors = [cmap(i) for i in np.linspace(0, 0.9, len(self.components.T['covariance']))]

        if (self.par_defs[par1].nuisance or  self.par_defs[par2].nuisance) and not self.use_nuisance:
            return False
        if (self.par_defs[par1].nuisance and self.par_defs[par2].nuisance) and self.no_nuisance_vs_nuisance:
            return False

        #plotting range
        ###
        x_min = self.par_defs[par1].min
        x_max = self.par_defs[par1].max
        y_min = self.par_defs[par2].min
        y_max = self.par_defs[par2].max

        if par1 in self.cuts:
            x_min = self.cuts[par1][0]
            x_max = self.cuts[par1][1]
        if par2 in self.cuts:
            y_min = self.cuts[par2][0]
            y_max = self.cuts[par2][1]

        ax = P.gca()

        # plot component means
        x_values = self.components.T['mean'].T[par1]
        y_values = self.components.T['mean'].T[par2]
        if centers:
            P.scatter(x_values[mask], y_values[mask], s=0.15)

        x_label = self.tr.to_tex(self.par_defs[par1].name)
        y_label = self.tr.to_tex(self.par_defs[par2].name)

        if x_label =="":
            x_label = "par"+str(par1)
        if y_label == "":
            y_label = "par"+str(par2)

        P.xlabel(x_label)
        P.ylabel(y_label)

        for i, c in enumerate(self.components.T['covariance']):
            #skip components by hand to retain consistent coloring
            if self.components.T['weight'][i] == 0.0: # or (self.single_chain is not None and i != int(self.single_chain)):
                continue

            # select a subrange of components
            if any(self.select) and (i < self.select[0] or i >= self.select[1]):
                continue

            cov = c.reshape((nCols, nCols))
            submatrix = np.array([[cov[par1,par1], cov[par1,par2]], \
                                  [cov[par2,par1], cov[par2,par2]]])

            # for idea, check
            # 'Combining error ellipses' by John E. Davis
            correlation = np.array([[1.0, cov[par1,par2] / np.sqrt(cov[par1,par1] * cov[par2,par2])], [0.0, 1.0]])
            correlation[1,0] = correlation[0,1]
            assert( -1 < correlation[0,1] and correlation[0,1] < 1)

            ew, ev = linalg.eigh(submatrix)
            assert(ew.min() > 0)

            # rotation angle of major axis with x-axis
            if submatrix[0,0] ==  submatrix[1,1]:
                theta = 0
            else:
                theta = 0.5 * np.arctan( 2 * submatrix[0,1] / (submatrix[1,1] - submatrix[0,0]))

            # put larger EW on y'-axis
            height = np.sqrt(ew.max())
            width = np.sqrt(ew.min())

            # but change orientation of coordinates if the other is larger
            if submatrix[0,0] > submatrix[1,1]:
                height = np.sqrt(ew.min())
                width = np.sqrt(ew.max())

            # change sign to rotate in right direction
            angle = - theta * 180 / np.pi

            # repeat spectrum multiple times
            if self.fold_color_spectrum > 0:
                color = colors[(i * self.fold_color_spectrum) % len(self.components.T['covariance'])]
            else:
                color = colors[i]

            # copy arguments
            kwargs_clone = dict(kwargs)

            # need full width/height
            e = Ellipse(xy=(x_values[i], y_values[i]), width=2*width, height=2*height, angle=angle, \
                        facecolor=kwargs.pop('facecolor', color), alpha=kwargs.pop('alpha', 0.3), **kwargs)
            ax.add_artist(e)

            if solid_edge:
                # important that some args already popped
                ax.add_artist(Ellipse(xy=(x_values[i], y_values[i]), width=2*width, height=2*height, angle=angle, \
                        facecolor='none', alpha=1, **kwargs))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        return True

    def proposal_weights(self):
        """
         Plot component weights
        """
        P.figure(figsize=(8,8))
        P.plot(self.components.T['weight'])
        P.xlabel('components')
        P.ylabel('weight')
        self.pdf_file.savefig()

    def compute_stats(self):
        """
        Compute perplexity and ESS
        """

        ###
        # normalize samples. Mask zero weights
        ###
        n = len(self.weights)
        w = self.weights / np.sum(self.weights)
        w = np.ma.MaskedArray(w, copy=False, mask=(w == 0))

        ###
        # perplexity. Avoid NaN due to log(0) by log(1)=0
        ###
        entr = - np.sum( w * np.log(w.filled(1.0)))
        perp = np.exp(entr) / n

        ###
        # ess
        ###
        coeff_var = np.sum((n * w - 1)**2) / n
        ess = 1.0 / (1.0 + coeff_var)

        print("Perplexity: %g, ESS: %g" % (perp, ess))
        return (perp, ess)

    def weight_distribution(self):
        """
        Plot histogram of sample weights
        """
        P.figure(figsize=(8,8))
        P.hist(self.weights, self.one_dim_n_bins)
        P.xlabel('weight')
        self.pdf_file.savefig()

    def plot(self):

        # output file name
        ext = ''
        if self.use_histogram:
            ext = '_hist'
        else:
            ext = '_KDE'

        if self.use_contours:
            ext += '_cont'

        # default mode: plot marginals
        one_dim = self.one_dimensional
        two_dim = self.two_dimensional
        epilog = lambda : 0 #self.weight_distribution

        # optionally draw pmc proposal instead
        if self.proposal and self.input_source == 'pmc':
            one_dim = lambda par : False
            two_dim = self.proposal_2D
            epilog = self.proposal_weights
            mask = self.components.T['weight'] > 0.0
            print("%d out of %d components still alive" % (len(np.where(mask)[0]), len(mask)))
            ext = '_prop'

        # plot only a few distributions if desired
        if self.single_1D is not None or self.single_2D is not None:
            if self.single_1D is not None:
                P.figure()
                one_dim(self.single_1D)
                name = self.__output_base_name + ext + "_par%d" % self.single_1D
                P.savefig(name + self.single_ext)

            if self.single_2D is not None:
                P.figure(figsize=(8,8))
                two_dim(self.single_2D[0], self.single_2D[1])
                name = self.__output_base_name + ext + "_par%d-par%d" % (self.single_2D[0], self.single_2D[1])
                P.savefig(name +  self.single_ext)

            return

        # default mode: plot all

        self.pdf_file_name = self.__output_base_name + ext + self.single_ext
        print("saving output to %s" % self.pdf_file_name)
        self.pdf_file = PdfPages(self.pdf_file_name)

        nCols =  len(self.par_defs)

        #take indices only from this list
        index_list = np.arange(nCols, dtype='int')

        #filter out nuisance parameters
        if not self.use_nuisance:
            scan_indices = []
            for i in index_list:
                if not self.par_defs[i].nuisance:
                    scan_indices.append(i)
            index_list = np.array(scan_indices)

        #1D marginals
        print("Plotting %d 1D marginal distributions" % len(index_list))
        for column in index_list:
                P.figure()
                if one_dim(column):
                    self.pdf_file.savefig()
                P.close()
        if self.one_dimensional_only:
            self.pdf_file.close()
            return

        #2D marginals
        print("Creating %d 2D marginal distributions, please stand by..." % int(len(index_list)*(len(index_list)-1)/2 ) )
        counter = 1
        for i, par1 in enumerate(index_list):
            for par2 in index_list[i + 1:]:

                #aspect ratio 1/1
                P.figure(figsize=(8,8))
                if two_dim(par1, par2) is not False:
                    print("plot #%d" % counter)
                    try:
                        P.tight_layout()
                    except harr:
                        pass
                    self.pdf_file.savefig()
                P.close()
                counter += 1

        # optional extra stuff
        epilog()

        self.pdf_file.close()

def factory(cmd_line=None):
    """
    Create the marginal distribution object from command line arguments
    """
    import argparse

    parser  = argparse.ArgumentParser(description='Plot marginal distributions of MCMC')
    parser.add_argument('i', metavar='input file',  help='HDF5 input file name')
    parser.add_argument('--1D-bins', help="Use fixed number of bins for 1D marginal distributions",action='store')
    parser.add_argument('--2D-bins', help="Use fixed number of bins for 2D marginal distributions",action='store')
    parser.add_argument('--1D-only', help="Plot only 1D marginal distributions",action='store_true')
    parser.add_argument('--bandwidth', help="Number in [0,1] used as bandwidth for KDE interpolation after rescaling to unit coordinate cube", action='store')
    parser.add_argument('--chains', help="Use only the specified chain for plotting, instead of all available chains", type=int, nargs='+')
    parser.add_argument('--contours', help="Add one and two sigma contours",action='store_true')
    parser.add_argument('--compute-stats', help="Compute perplexity and effective sample size (PMC only)", action='store_true')
    parser.add_argument('--cut', help="Add a cut for computing the integral: PAR MIN MAX",nargs=3, action='append')
    parser.add_argument('--evolution', help="Plot the evolution  of individual chains, either on the 'log' or 'linear' scale", action='store', default='harr')
    parser.add_argument('--integrate', help="Compute integral over hyperrectangle, use with --cut",action='store_true')
    parser.add_argument('--comp-integrate', help="Compute evidence from individual components",action='store_true')
    parser.add_argument('--hc-initial', help="Read initial guess from long patches for hierarchical clustering",action='store_true')
    parser.add_argument('--hc-patches', help="Read components from short chain patches for hierarchical clustering",action='store_true')
    parser.add_argument('--min-prob', help="Whiten all bins with prob less than this value",action='store')
    parser.add_argument('--nuisance', help="Plot nuisance parameters.", action='store_true', default=False)
    parser.add_argument('--no-nuisance-vs-nuisance', help="Don't produce 2D plots if both are nuisance parameters",action='store_true', default=False)
    parser.add_argument('--prior', help="Plot the prior in 1D distributions.",action='store_true')
    parser.add_argument('--single-1D', help="Plot only the 1D marginal distribution of the ith parameter, i=0...N-1", action='store')
    parser.add_argument('--single-2D', help="Plot only the 2D marginal distribution of parameters i,j, i<j, i,j=0...N-1", nargs=2)
    parser.add_argument('--single-ext', help="File extension for single plots, e.g 'pdf'[default] or 'png'", action='store')
    parser.add_argument('--select', help="Select a range of samples from each chain", action='store',nargs=2, default=(None, None))
    parser.add_argument('--skip-initial', help="Allows to skip the first fraction of iterations", action='store', default=0)
    parser.add_argument('--output-dir', help="Store plots in different directory than input file", action='store')
    parser.add_argument('--mcmc', help="Treat input as file from mcmc", action='store_true')
    parser.add_argument('--nest', help="Treat input as file from multinest", action='store_true')
    parser.add_argument('--pmc-crop-outliers', help="Remove N points with the highest weight", action='store', default=0)
    parser.add_argument('--pmc-equal-weights', help="Plot PMC proposal function by giving each drawn samples the same weight", action='store_true')
    parser.add_argument('--pmc-proposal', help="Plot PMC proposal function", action='store_true')
    parser.add_argument('--pmc-stats', help="Plot evolution of convergence diagnostics and evidence", action='store_true')
    parser.add_argument('--pmc-step', help="Use a specified step, default: final step", action='store')
    parser.add_argument('--pmc-queue-output', help="Treat input as file from PMC queue manager", action='store_true', default=False)
    parser.add_argument('--prerun', help="Use prerun instead of main", action='store_true')
    parser.add_argument('--gof', help="Determine GoF for a particular point. Specify each coordinate independently as --gof i value, e.g. --gof 0 0.4 --gof 1 0.8 i<j, i,j=0...N-1", action='append', nargs=2)
    parser.add_argument('--use-data-range', help="Determine the parameter ranges from data, instead of from definition in HDF5. ", action='store', default=0.0)
    parser.add_argument('--use-KDE',  help='Use kernel density estimation instead of histograms', action='store_true')

    # defaults to sys.argv if None is passed in
    args = parser.parse_args(cmd_line)

    ###
    # setup the object
    ###

    # determine input mode
    input_source_default = 'pmc'
    input_source = input_source_default
    if args.mcmc:
        input_source = 'mcmc'
    if args.nest:
       assert(input_source == input_source_default)
       input_source = 'multinest'

    if args.hc_initial:
        hc_comp = 'long'
    elif args.hc_patches:
        hc_comp = 'short'
    else:
        hc_comp = None

    #had some trouble getting the second argument out-of the namepace, so use the dictionary directly
    marg = MarginalDistributions(args.i, args.__dict__['use_KDE'], output_dir=args.__dict__['output_dir'],
                                 input_source=input_source, pmc_queue_output=args.pmc_queue_output,
                                 pmc_crop_outliers=float(args.__dict__['pmc_crop_outliers']),
                                 pmc_equal_weights=args.__dict__['pmc_equal_weights'] or args.pmc_proposal,
                                 chains=args.chains or args.__dict__['pmc_step'], prerun=args.__dict__['prerun'],
                                 nuisance=args.nuisance, select=[int(x) if x is not None else x for x in args.__dict__['select']],
                                 skip_initial=float(args.__dict__['skip_initial']), hc_comp=hc_comp)
    if  args.__dict__['use_KDE']:
        marg.kde_reduction = 1
    if  args.__dict__['use_data_range'] > 0:
        marg.use_data_range = True
        marg.scale_data_range = float(args.__dict__['use_data_range'])
    if  args.__dict__['bandwidth'] is not None:
        marg.kde_bandwidth = float(args.__dict__['bandwidth'])
    if args.__dict__['1D_bins'] is not None:
        marg.fixed_1D_binning = True
        marg.one_dim_n_bins = args.__dict__['1D_bins']
    if args.__dict__['2D_bins'] is not None:
        marg.fixed_2D_binning = True
        marg.two_dim_n_bins = args.__dict__['2D_bins']
    if args.__dict__['1D_only']:
        marg.one_dimensional_only = True
    if args.min_prob is not None:
        marg.minimum_probability = float(args.min_prob)
    if args.nuisance:
        marg.no_nuisance_vs_nuisance = args.no_nuisance_vs_nuisance
        marg.nuisance_2D = False if args.nuisance == '1D' else True
    if args.__dict__['contours']:
        marg.use_contours = True
    if args.__dict__['cut']:
        print("Cuts passed by command line arguments: %d " % len(args.__dict__['cut']))
        for cut in args.__dict__['cut']:
            cut[0] = int(cut[0])
            if cut[1] == 'MIN':
                cut[1] = marg.par_defs[cut[0]].min
            if cut[2] == 'MAX':
                cut[2] = marg.par_defs[cut[0]].max
            marg.cuts[cut[0]] = (float(cut[1]), float(cut[2]))
    if args.__dict__['prior']:
        marg.plot_prior = True
    if args.__dict__['pmc_proposal']:
        marg.proposal = True
    if args.__dict__['single_1D'] is not None:
        marg.single_1D = int(args.__dict__['single_1D'])
        marg.use_nuisance = True
    if args.__dict__['single_2D'] is not None:
        marg.single_2D = [int(x) for x in args.__dict__['single_2D']]
        marg.no_nuisance_vs_nuisance = False
    if args.__dict__['single_ext']:
        marg.single_ext = "." + args.__dict__['single_ext']
    if args.__dict__['output_dir'] is not None:
        marg.output_dir =  args.__dict__['output_dir']
    if args.__dict__['gof'] is not None:
        for pair in args.__dict__['gof']:
            print(pair)
            marg.gof_point[int(pair[0])] = float(pair[1])

    if cmd_line is not None:
        return marg

    ### do the mutually exclusive work
    if args.__dict__['evolution'] is not 'harr':
        print("Plotting evolution")
        marg.evolution(args.__dict__['evolution'])
    elif args.__dict__['integrate']:
        marg.integrate()
    elif args.__dict__['comp_integrate']:
        marg.comp_integrate()
    elif args.__dict__['compute_stats']:
        marg.compute_stats()
    elif args.__dict__['pmc_stats']:
        marg.convergence()
    else:
        marg.plot()

def test_ellipse():
    from matplotlib.patches import Ellipse

    P.figure(figsize=(8,8))
    ax = P.gca()

    submatrix = np.array([[0.3**2   , -0.1], \
                          [-0.1   , 0.4**2]])

    P.xlim((4.5, 5.5))
    P.ylim((0, 1))

    ew, ev = linalg.eigh(submatrix)

    aspect_ratio = 1
    theta = 0.5 * np.arctan( 2 * submatrix[0,1] / (submatrix[1,1] - submatrix[0,0]))

    # put larger EW on y'-axis
    height = np.sqrt(ew.max())
    width = np.sqrt(ew.min())

    if submatrix[0,0] > submatrix[1,1]:
        height = np.sqrt(ew.min())
        width = np.sqrt(ew.max())

    print(ew)

    # change sign to rotate in right direction
    angle = - theta * 180 / np.pi

    # need full width/height
    e = Ellipse(xy=(5, 0.5), width=2*width, height=2*height, angle=angle)
    ax.add_artist(e)
    P.show()
    P.savefig('ellipse.pdf')

def main():

    # do all the plotting
    factory()

if __name__ == '__main__':
    np.set_printoptions(precision=6)
    matplotlib.rcParams['text.latex.unicode'] = True

    main()
