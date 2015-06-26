#! /usr/bin/python
"""
Plot several distributions with their contours for paper
"""
from __future__ import print_function

import matplotlib
import matplotlib.ticker as ticker
import matplotlib.patches
import matplotlib.pyplot as P
from matplotlib.lines import Line2D
import numpy as np
import os, sys
import itertools
from collections import OrderedDict, defaultdict, namedtuple
from sets import Set

sys.path.append(os.path.realpath('../plot'))
import plotScript
from plotScript import ParameterDefinition


def run_command(com):
    import commands

    status, output = commands.getstatusoutput(com)
    if status != 0:
        raise Exception(output)

    return output


def square_figure(size=6):
    return P.figure(figsize=(size, size))


def wide_figure(x_size=8, ratio=4 / 3.0, left=0.165, right=0.95, top=0.95, bottom=0.145):
    """
    Create a figure with aspect ratio 4:3 and sufficient room for margins
    """
    y_size = x_size / ratio
    fig = P.figure(figsize=(x_size, y_size))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=left)
    fig.subplots_adjust(right=right)
    fig.subplots_adjust(top=top)
    fig.subplots_adjust(bottom=bottom)

    return ax


def adjust_subplot(fig=None, equal=True):
    # 1:1 aspect ratio, but only if called before adjusting edges
    if equal:
        P.gca().set_aspect('equal')

    if fig is None:
        fig = P.gcf()
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.15)


class Scenario(object):
    def __init__(self, file, color, nbins=200, alpha=0.4, sigma='2 sigma',
                 bandwidth_default=None, two_sigma_color=None, queue_output=True,
                 crop_outliers=200, local_mode=None, defs={},
                 unc_label='', histogram=True, linestyle='solid'):
        self.f = file
        self.c = color
        self.prob = {}
        self.nbins = nbins
        self.alpha = alpha
        self.sigma = sigma
        self.__bandwidth = {}
        self.__bandwidth_default = bandwidth_default
        self.two_sigma_color = two_sigma_color
        self.queue_output = queue_output
        self.crop_outliers = crop_outliers
        self.local_mode = local_mode
        # dictionary of name:ParameterDefinition
        self.defs = defs
        self.unc_label = unc_label
        self.histogram = histogram
        self.linestyle = linestyle

    def get_bandwidth(self, par1, par2):
        try:
            return self.__bandwidth[(par1, par2)]
        except KeyError:
            return self.__bandwidth_default

    def get_bandwidth1D(self, par1):
        try:
            return self.__bandwidth[par1]
        except KeyError:
            return self.__bandwidth_default

    def set_bandwidth(self, par1, par2, value):
        self.__bandwidth[(par1, par2)] = value

    def set_bandwidth1D(self, par1, value):
        self.__bandwidth[par1] = value


class Measurement(object):
    '''Store info to plot x +a -b for a measurement.'''

    def __init__(self, lower, central, upper, central_style=dict(), one_sigma_style=None, label=''):
        self.central = central
        self.lower = lower
        self.upper = upper
        self.central_style = central_style
        self.one_sigma_style = one_sigma_style
        if one_sigma_style is None:
            self.one_sigma_style = self.central_style
        self.label = label


class ScenarioComparison(object):
    """Store parameter definitions and bandwidths to compare
    the marginals of two scenarios.

    Parameters
    ----------

    defs: sequence
    Could be a sequence of (pairs of) ParameterDefinition. Pairs would be
    used for a comparison of individual modes
    """

    def __init__(self, defs, bandwidths):
        self.defs = defs
        self.bandwidths = bandwidths

        """
                # store bandwidths for comparison of two scenarios
        # format: (scenario, i, j, par1, par2)
        self.comparison_bandwidths = {}

        self.comparison_defs = {}

        name = 'Re{c7}'
        nticks = 5
        par_def1 = ParameterDefinition(name=name, min=0.2, max=0.6, index=self.pars.index(name))
        par_def1.major_locator = ticker.FixedLocator(np.linspace(par_def1.min, par_def1.max, nticks))
        par_def2 = ParameterDefinition(name=name, min=-0.5, max=-0.1, index=self.pars.index(name))
        par_def2.major_locator = ticker.FixedLocator(np.linspace(par_def2.min, par_def2.max, nticks))
        self.comparison_defs[name] = (par_def1, par_def2)

        self.comparison_defs['Re{c9}'] = (ParameterDefinition(name='Re{c9}', min=+1, max=+6, index=self.pars.index('Re{c9}')),
                                ParameterDefinition(name='Re{c9}', min=-7, max=-2, index=self.pars.index('Re{c9}')))
        self.comparison_defs['Re{c10}'] = (ParameterDefinition(name='Re{c10}', min=+1.5, max=+6.5, index=self.pars.index('Re{c10}')),
                                 ParameterDefinition(name='Re{c10}', min=-6.5, max=-1.5, index=self.pars.index('Re{c10}')))
        """


class MarginalContours(object):
    def __init__(self, input_base, output_base, ext='.pdf', max_samples=None, ignore_scenarios=None):

        self.output_base = os.path.join(output_base, 'paper_plots')
        if not os.path.isdir(self.output_base):
            os.mkdir(self.output_base)
        self.ext = ext

        self.input_base = input_base

        self.scen = OrderedDict()
        # SM prediction for C7, C9, C10
        self.sm_point = {'Re{c7}': -0.33726473, 'Re{c9}': +4.27342842, 'Re{c10}': -4.16611761}
        self.sm_point_style = dict(marker='D', markersize=8, color='black')

        best_fit_point_style = dict(self.sm_point_style)
        best_fit_point_style['marker'] = 'x'
        best_fit_point_style['markeredgewidth'] = 3.0

        next_best_fit_point_style = dict(best_fit_point_style)
        next_best_fit_point_style['marker'] = '+'
        next_best_fit_point_style['color'] = 'Blue'

        self.best_fit_points_style = [[best_fit_point_style] * 4, [next_best_fit_point_style] * 4]

        # highest to lowest contour
        self.contour_styles = [dict(linestyle='solid'),
                               dict(linestyle='dashed'),
                               dict(linestyle='dashdot')]

        self.contour_alpha = (0.8, 0.5, 0.3)

        self.max_samples = max_samples
        self.read_data()

        # store expensive calculation of contours
        self.__density_cache = {}

        # ranges and bandwidths for scenario comparison
        self.comparison_bandwidths = {}
        self.comparison_defs = {}

    def out(self, name):
        """ Create output file name"""
        return os.path.join(self.output_base, name) + self.ext

    ###
    # template for single plots
    ###
    def command_template(self, scenario):
        cmd_template = ''
        # input file
        cmd_template += ' %s' % scenario.f
        if self.max_samples is not None:
            cmd_template += ' --select 0 %d' % self.max_samples
        cmd_template += ' --contours  --pypmc'
        if 'mcmc' in scenario.f:
            cmd_template += ' --mcmc --skip-init 0.2'
            cmd_template += ' --2D-bins %d' % scenario.nbins
        elif 'unc' in scenario.f:
            cmd_template += ' --unc'
            cmd_template += ' --1D-bins %d' % scenario.nbins
            cmd_template += ' --use-data-range 0.05'
        elif 'vb' in scenario.f:
            cmd_template += ' --pmc-crop-outliers %d' % scenario.crop_outliers
            cmd_template += ' --2D-bins %d' % scenario.nbins
            cmd_template += ' --determ'
        else:
            raise Exception('Unknown file type' + scenario.f)
        if not scenario.histogram:
            cmd_template += ' --use-KDE'

        if not 'unc' in scenario.f:
            for p in scenario.defs:
                cmd_template += ' --cut %d %s %s' % (p.i, p.min, p.max)

        return cmd_template.split()

    def read_data(self):

        self.margs = {}
        for k in self.scen.keys():
            self.margs[k] = plotScript.factory(self.command_template(self.scen[k]))

    def compute_marginal1D(self, def1, scenario):
        i = def1.i
        m = self.margs[scenario]
        if m.marginal_modes.has_key(i):
            return

        bandwidth = self.scen[scenario].get_bandwidth1D(i)
        if m.use_histogram and bandwidth is None:
            pass
        else:
            m.use_histogram = False
            if bandwidth:
                m.kde_bandwidth = bandwidth

        # if def1.min != def1.max:
        #     m.cuts[i] = def1.range
        m.one_dimensional(i)
        assert len(m.credibilities[0][0]) == 1, "Found disconnected 1sigma interval\n" + str(m.credibilities[0][0])

    def compute_marginal1D_all(self, scenarios):
        for scenario in scenarios:
            for def1 in self.margs[scenario].out.par_defs:
                self.compute_marginal1D(def1, scenario)

    def compute_marginal2D(self, def1, def2, scenario, solution=0):
        # parameter index
        i = def1.i
        j = def2.i

        if self.__density_cache.has_key((i, j, scenario, solution)):
            return

        m = self.margs[scenario]
        bandwidth = self.comparison_bandwidths.get((scenario, solution, def1.name, def2.name), None)
        if bandwidth is None:
            bandwidth = self.scen[scenario].get_bandwidth(i, j)
        if bandwidth is None:
            m.use_histogram = True
        else:
            m.use_histogram = False
            m.kde_bandwidth = bandwidth

        m.cuts[i] = def1.range
        m.cuts[j] = def2.range

        # compute the density
        density = m.two_dimensional(i, j)

        # store the density and the associated ranges
        if not np.iterable(solution):
            solution = (solution,)
        for sol in solution:
            self.__density_cache[(i, j, scenario, sol)] = (density, def1.range, def2.range)

        # clean up
        P.cla()

    def single_panel(self, def1, def2, scenarios,
                     solution=None, SM_point=True, local_mode=[True],
                     desired_levels=None, label=True, indices=None):
        """
        Single marginal plot to compare multiple scenarios


        :param solution:
            Index to identify a region of zoom (a *solution*) into the 2D marginal.

        :param indices:
            Indices of parameters in file. Example: [(0, 1), (23, 47)]

        """

        # assert (len(scenarios) in (1, 2))
        if indices:
            par_indices_i = [ind[0] for ind in indices]
            par_indices_j = [ind[1] for ind in indices]
        else:
            print(def1.name, def2.name)
            # get parameter indices in EOS output file by parameter name
            par_indices_i = []
            par_indices_j = []
            for s in scenarios:
                m = self.margs[s]
                # search for parameter by name
                for n, p in enumerate(m.out.par_defs):
                    if p.name == def1.name:
                        par_indices_i.append(n)
                    elif p.name == def2.name:
                        par_indices_j.append(n)
        assert len(par_indices_i) == len(scenarios), par_indices_i
        assert len(par_indices_j) == len(scenarios), par_indices_j

        # compute and cache densities
        for n, s in enumerate(scenarios):
            # tmp store
            i, j = def1.i, def2.i
            def1.i, def2.i = par_indices_i[n], par_indices_j[n]
            self.compute_marginal2D(def1, def2, s, solution)
            # rewind
            def1.i, def2.i = i, j

        for k, s in enumerate(scenarios):
            # retrieve original range used to create prob_density
            # then zoom will work correctly
            density, xrange, yrange = self.__density_cache[(par_indices_i[k], par_indices_j[k], s, solution)]
            for line in [None, self.scen[s].linestyle]:
                artist = self.margs[s].contours_two(xrange, yrange, density,
                                                    color=self.scen[s].c if line is None else 'black', line=line, grid=True,
                                                    desired_levels=desired_levels,
                                                    alpha=self.contour_alpha if line is None else 0.8)

        if SM_point:
            P.plot(self.sm_point.get(def1.name, 0.), self.sm_point.get(def2.name, 0.), **self.sm_point_style)
        if any(local_mode):
            for n, s in enumerate(scenarios):
                if local_mode[n]:
                    for style, p in zip(self.best_fit_points_style[n], self.scen[s].local_mode):
                        P.plot(p[par_indices_i[n]], p[par_indices_j[n]], **style)

        ax = P.gca()

        if hasattr(def1, 'major_locator'):
            ax.xaxis.set_major_locator(def1.major_locator)
        # zoom in
        ax.set_xlim(def1.range)

        if hasattr(def2, 'major_locator'):
            ax.yaxis.set_major_locator(def2.major_locator)
        # zoom in
        ax.set_ylim(def2.range)

        if label:
            P.xlabel(self.margs[scenarios[0]].tr.to_tex(def1.name))
            P.ylabel(self.margs[scenarios[0]].tr.to_tex(def2.name))

    def stack(self, par1, par2, scenarios, solution=(1, 0), **kwargs):
        """
        Plot two panels, one on top of the other with
        marginal distribution for two parameters. Each panel
        focuses on one solution/region of the target.

        Keyword arguments:
        par1, par2 -- parameter names.
        solution -- specify indices of solution. First solution on bottom, second on top.
        """

        # choose x_size such that fonts are readible
        # y size manually adjusted so actual plot frame (not the figure!) has 1:1 aspect ratio
        x_size = 5
        fig = P.figure(figsize=(x_size, 1.8 * x_size))

        for i, s in enumerate(solution):
            P.subplot(2, 1, i)
            def1 = self.comparison_defs[(par1, s)]
            def2 = self.comparison_defs[(par2, s)]
            self.single_panel(def1, def2, solution=s, scenarios=scenarios, label=False, **kwargs)

        # Set common labels
        tr = self.margs[scenarios[0]].tr
        fig.text(0.56, 0.02, tr.to_tex(def1.name), ha='center', va='center')
        fig.text(0.04, 0.53, tr.to_tex(def2.name), ha='center', va='center', rotation='vertical')

        fig.subplots_adjust(left=0.2, right=0.92, bottom=0.1, top=0.98)

    def compare_scenarios(self, combinations, **kwargs):
        """
        compare multiple scenarios in overlays with filled and contoured regions
        """

        for c in combinations:
            self.stack('Re{c7}', 'Re{c9}', scenarios=c, **kwargs)
            P.savefig(self.out("%s_0_1" % '_'.join(c)))

            self.stack('Re{c7}', 'Re{c10}', scenarios=c, **kwargs)
            P.savefig(self.out("%s_0_2" % '_'.join(c)))

            self.stack('Re{c9}', 'Re{c10}', scenarios=c, **kwargs)
            P.savefig(self.out("%s_1_2" % '_'.join(c)))

    def overlays(self):
        ###
        # create overlays
        ###
        P.figure(figsize=(8, 8))

        for i, p1 in enumerate(self.pars):
            for j, p2 in enumerate(self.pars[self.pars.index(p1) + 1:]):
                P.clf()
                # draw SM point
                P.plot(self.sm_point[self.defs[p1].i], self.sm_point[self.defs[p2].i], **self.sm_point_style)

                for k in self.scen.keys():
                    P.xlabel(self.margs[k].tr.to_tex(p1))
                    P.ylabel(self.margs[k].tr.to_tex(p2))
                    # draw contours
                    CS = self.margs[k].contours_two(self.defs[p1].range, self.defs[p2].range,
                                                    self.scen[k].prob[(p1, p2)],
                                                    desired_levels=self.scen[k].sigma, color=self.scen[k].c, grid=False)
                    P.setp(CS.collections[1], alpha=self.scen[k].alpha)
                    c = self.scen[k].two_sigma_color
                    if c is not None:
                        P.setp(CS.collections[1], color=c)

                    # don't whiten beyond 2 sigma, so ignore lowest contour fill
                    # P.setp(CS.collections[0], alpha=0.0)
                    CS.collections[0].remove()

                    P.savefig(self.out('overlay_%d_%d_%s' % (i, self.pars.index(p1) + j + 1, k)))

    def prediction(self, scen, one_sigma_style={}, two_sigma_style=None):
        '''Plot uncertainty band for one observable'''
        m = self.margs[scen]

        # compute uncertainties
        lower_one_sigma = np.empty(len(m.out.par_defs))
        upper_one_sigma = np.empty_like(lower_one_sigma)
        lower_two_sigma = np.empty_like(lower_one_sigma)
        upper_two_sigma = np.empty_like(lower_one_sigma)
        mode = np.empty_like(lower_one_sigma)

        for def1 in m.out.par_defs:
            lower_one_sigma[def1.i] = m.credibilities[def1.i][0][0, 0]
            upper_one_sigma[def1.i] = m.credibilities[def1.i][0][0, 1]
            lower_two_sigma[def1.i] = m.credibilities[def1.i][1][0, 0]
            upper_two_sigma[def1.i] = m.credibilities[def1.i][1][0, 1]

        # plot every one
        if two_sigma_style is not None:
            P.fill_between(m.out.fixed_values, lower_two_sigma, upper_two_sigma, **two_sigma_style)
        P.fill_between(m.out.fixed_values, lower_one_sigma, upper_one_sigma,
                       **one_sigma_style)

    def stack_prediction(self, scenarios, measurement, legendpos='best'):
        legend_patches = []

        # plot measurement first and below the patches
        m = self.margs[scenarios[0]]
        x_range = m.out.fixed_values[0], m.out.fixed_values[-1]
        if measurement:
            P.fill_between(x_range, [measurement.lower] * 2, [measurement.upper] * 2, **measurement.one_sigma_style)
            P.plot(x_range, [measurement.central] * 2, **measurement.central_style)
            legend_patches.append((matplotlib.patches.Patch(**measurement.one_sigma_style), measurement.label))

        for scen in scenarios:
            s = self.scen[scen]
            one_sigma_style = dict(alpha=0.6, color=s.c)
            two_sigma_style = dict(one_sigma_style)
            two_sigma_style['alpha'] = 0.3
            self.prediction(scen, one_sigma_style, two_sigma_style)
            legend_patches.append((matplotlib.patches.Patch(color=one_sigma_style['color'], alpha=one_sigma_style['alpha']), s.unc_label))

        # set up legend manually, labels not supported by fill_between
        # http://stackoverflow.com/q/14534130/987623
        patches = [p[0] for p in legend_patches]
        labels = [p[1] for p in legend_patches]
        P.legend(patches, labels, loc=legendpos, fontsize=matplotlib.rcParams['font.size'] - 2)

        P.xlim(x_range)


def input_output():
    try:
        input_base = os.environ['EOS_RESULTS']
    except KeyError:

        msg = """\
Set EOS_RESULTS to base directory (e.g. with symlinks)
that contains the final results
Example:
~/data/eos$ ll summer2013/
total 456K
lrwxrwxrwx 1 beaujean beaujean   56 Jul 12 13:09 pmc_final_scI_all.hdf5 -> ../2013-07-09/scI_all/pmc_parameter_samples_5.hdf5_merge
"""
        print >> sys.stderr, msg
        raise

    output_base = input_base

    return (input_base, output_base)


def transform(data, i, j):
    '''Assume `data` is a matrix. Modify `data` in-place to contain the sum and difference of the two columns `i, j` in `data.'''

    # pick out two nonadjacent columns with a slice object to return a view(!)
    data = data[:, i:j + 1:j - i]
    buffer = np.array(data.T[1])
    data.T[1][:] = data.T[0]
    data.T[0] += buffer
    data.T[1] -= buffer


def betrag(data, i, j):
    '''Compute the Betrag of complex number, real part in column `i`, imaginary part in column `j` into column `k`.'''
    buffer = data.T[i] * data.T[i]
    buffer += data.T[j] * data.T[j]
    return np.sqrt(buffer)


class Spring2015(object):
    # set up object
    input_base, output_base = input_output()
    max_samples = None
    fig_size = 6  # inches, same for x and y

    def input(self, file):
        return os.path.join(self.input_base, file)

    def overlay(self, marg, def1, def2, scenarios, desired_levels=(0.683, 0.954), indices=None):
        marg.single_panel(def1, def2, scenarios=scenarios,
                          desired_levels=desired_levels,
                          local_mode=[False] * len(scenarios),
                          indices=indices)  # no local mode
        adjust_subplot()
        P.savefig(marg.out(','.join(scenarios) + '_%d,%d' % (def1.i, def2.i)))

    def figSP(self):

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        ###
        # scenarios
        ###

        s = 'scSP_FH'
        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_' + s + '.hdf5'), '#848484',
                                nbins=300, linestyle='dotted')
        #                                         local_mode=[[-0.348441713,   3.787226592, -4.420530192],
        #                                                     [ 0.5021320352, -4.568457245,  4.25129282]])
        marg.scen[s].set_bandwidth(0, 2, 0.025)
        marg.scen[s].set_bandwidth(0, 4, 0.04)
        marg.scen[s].set_bandwidth(4, 6, 0.025)

        # indices of S,S' and P,P'
        marg.scen[s].spind = [(0, 2), (4, 6)] # real
        # marg.scen[s].spind = [(1, 3), (5, 7)] # imaginary

        s = 'scSP_Bsmumu'
        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_' + s + '.hdf5'), '#2E64FE',
                                nbins=300, linestyle='dashed')
        #                                         local_mode=[[-0.348441713,   3.787226592, -4.420530192],
        #                                                     [ 0.5021320352, -4.568457245,  4.25129282]])
        marg.scen[s].set_bandwidth(0, 2, 0.015)
        marg.scen[s].set_bandwidth(2, 6, 0.008)
        marg.scen[s].set_bandwidth(4, 6, 0.015)

        # indices of S,S' and P,P'
        marg.scen[s].spind = [(0, 2), (4, 6)] # real
        # marg.scen[s].spind = [(1, 3), (5, 7)] # imaginary

        s = 'sc910SP_K_KstarBR_Bsmumu'
        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'vb_' + s + '.hdf5'), '#FE2E2E',
                                nbins=300, crop_outliers=500)
        # indices of S,S' and P,P'
        marg.scen[s].spind = [(8, 10), (12, 14)] # real
        # marg.scen[s].spind = [(9, 11), (13, 15)] # imaginary

        marg.scen[s].set_bandwidth(marg.scen[s].spind[0][0], marg.scen[s].spind[0][1], 0.025) # +-
        marg.scen[s].set_bandwidth(marg.scen[s].spind[0][0], marg.scen[s].spind[1][0], 0.055) # ++
        marg.scen[s].set_bandwidth(marg.scen[s].spind[0][1], marg.scen[s].spind[1][1], 0.015) # --
        marg.scen[s].set_bandwidth(marg.scen[s].spind[1][0], marg.scen[s].spind[1][1], 0.025) # +- P

        marg.read_data()
        scenarios = marg.scen.keys()

        ###
        # transformations
        ###
        for k, v in marg.scen.iteritems():
            m = marg.margs[k]
            data = m.out.samples
            transform(data, *v.spind[0])
            transform(data, *v.spind[1])

        ###
        # plot settings
        ###
        defs = [ParameterDefinition(index=0, name=r"$\mathrm{Re}(\mathcal{C}_S + \mathcal{C}_{S'})$", min=-1, max=1),
                ParameterDefinition(index=2, name=r"$\mathrm{Re}(\mathcal{C}_S - \mathcal{C}_{S'})$", min=-1, max=1),
                ParameterDefinition(index=4, name=r"$\mathrm{Re}(\mathcal{C}_P + \mathcal{C}_{P'})$", min=-1, max=1),
                ParameterDefinition(index=6, name=r"$\mathrm{Re}(\mathcal{C}_P - \mathcal{C}_{P'})$", min=-1, max=1),
                ]

        square_figure(self.fig_size)
        matplotlib.rcParams['font.size'] = 20

        # 1D intervals for table
        '''
        for k, v in marg.scen.iteritems():
            m = marg.margs[k]

            print("scenario", k)
            for i, d in enumerate(defs):
                P.clf()
                m.nBins[-1] = 200
                ind_in_file = v.spind[i // 2][i % 2]
                m.fixed_1D_binning = False
                m.use_histogram = False
                m.kde_bandwidth = 0.05
                m.nBins[ind_in_file] = 200

                m.one_dimensional(ind_in_file)
                P.savefig(marg.out(k + '_Betrag_' + d.name))
        return
        '''
        # + vs - S
        self.overlay(marg, defs[0], defs[1], scenarios, indices=[marg.scen[s].spind[0] for s in scenarios])

        # + vs +
        sel = ['scSP_FH', 'sc910SP_K_KstarBR_Bsmumu']
        self.overlay(marg, defs[0], defs[2], sel,
                     indices=[(marg.scen[s].spind[0][0], marg.scen[s].spind[1][0]) for s in sel])

        # - vs -
        sel = ['scSP_Bsmumu', 'sc910SP_K_KstarBR_Bsmumu']
        self.overlay(marg, defs[1], defs[3], sel,
                     indices=[(marg.scen[s].spind[0][1], marg.scen[s].spind[1][1]) for s in sel])

        # + vs - P
        self.overlay(marg, defs[2], defs[3], scenarios, indices=[marg.scen[s].spind[1] for s in scenarios])


    def figTT5(self):

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        ###
        # scenarios
        ###
        s = 'scTT5_FH'
        scenarios = [s]
        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_scTT5_FH.hdf5'), 'OrangeRed',
                                nbins=300)
        # in python, we can add class members as we please, use with care!
        # indices of real and imaginary part
        marg.scen[s].ctind = [(0, 1), (2, 3)]

        s = 'scTT5_K_KstarBR'
        scenarios.append(s)
        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_' + s + '.hdf5'), 'Blue',
                                nbins=300)
        marg.scen[s].ctind = [(0, 1), (2, 3)]

        s = 'sc910TT5_K_KstarBR_Bsmumu'
        scenarios.append(s)
        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_' + s + '.hdf5'), 'Green',
                                nbins=300)
        marg.scen[s].ctind = [(4, 5), (6, 7)]

        marg.read_data()

        ###
        # plot absolute value of CT and CT5
        ###
        for s in scenarios:
            m = marg.margs[s]
            data = m.out.samples
            for ii, name in zip(marg.scen[s].ctind, ['CT', 'CT5']):
                data[:, -1] = betrag(data, ii[0], ii[1])
                m.out.par_defs[-1].nuisance = False
                m.out.par_defs[-1].min = 0.0
                m.out.par_defs[-1].max = 1.45
                m.use_contours = True
                m.fixed_1D_binning = False
                m.nBins[-1] = 200

                P.clf()
                m.one_dimensional(data.shape[1] - 1)
                P.savefig(marg.out(s + '_Betrag_' + name))

    def fig_pred(self, scen_obs, measurement, ylabel, xlabel=r'$C_{T}$', yrange=None,
                 legendpos='upper center', rescale=None):
        '''Plot predictions/postdictions with uncertainty varying one parameter over a fixed range.'''

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        scen_kw = dict(nbins=200, file_type='unc', histogram=False)
        colors = ['blue', 'red']
        scenario_names = []
        for i, d in enumerate(scen_obs):
            kw = dict(scen_kw)
            kw['color'] = colors[i]
            name = d.pop('name')
            kw.update(d)
            marg.scen[name] = Scenario(self.input('unc_' + name + '.hdf5'), **kw)
            scenario_names.append(name)
        marg.read_data()

        # rescale every sample by a constant
        # this requires updating cached min/max values, too.
        if rescale:
            for k, v in marg.margs.iteritems():
                v.out.samples *= rescale
                v.min *= rescale
                v.max *= rescale

        marg.compute_marginal1D_all(scenario_names)
        wide_figure(x_size=8, ratio=4 / 3.0, left=0.165, right=0.95, top=0.93, bottom=0.145)

        central_style = dict(color='black', linewidth=matplotlib.rcParams['axes.linewidth'],
                             linestyle='solid', alpha=1)
        one_sigma_style = dict(color='grey', alpha=1)
        if measurement:
            if not measurement.has_key('sigma_lower'):
                measurement['sigma_lower'] = measurement['sigma_upper']
            measurement = Measurement(label=r'$\mathrm{LHCb}$',
                                      central_style=central_style,
                                      one_sigma_style=one_sigma_style,
                                      central=measurement['central'],
                                      lower=measurement['central'] - measurement['sigma_lower'],
                                      upper=measurement['central'] + measurement['sigma_upper'])

        marg.stack_prediction(scenario_names, measurement,
                              legendpos=legendpos)
        P.xlabel(xlabel)
        P.ylabel(ylabel)

        if yrange:
            P.ylim(*yrange)
        P.savefig(marg.out(scenario_names[0]))

    def np_label(self, value):
        return r'$\mathcal{C}_9^{\rm NP} = ' + str(value) + '$'

    def scenario_comparison(self, obs, s_min, s_max, scen=('ct', 'ct_c9_1dot1'), delta_c9=(0, -1.1)):
        return [dict(name=scen[i] + '_' + obs + '%gto%g' % (s_min, s_max), unc_label=self.np_label(delta_c9[i])) for i
                in range(len(scen))]

    def fig_1(self):
        matplotlib.rcParams['font.size'] = 28

        obs = 'K_FH'
        ylabel = r'$\langle F_H \rangle'
        yrange = (-0.05, 0.9)

        s_min, s_max = 1, 6
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=dict(central=0.03, sigma_upper=0.036),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        s_min, s_max = 15, 22
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=dict(central=0.035, sigma_upper=0.04031),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        obs = 'K_BR'
        yrange = (0.5e-7, 2.75e-7)
        ylabel = r'$\langle \mathcal{B}(K) \rangle'

        s_min, s_max = 1, 6
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=dict(central=2.42e-8 * 4.9, sigma_upper=6.8072975548304046e-09),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        s_min, s_max = 15, 22
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=dict(central=1.21e-8 * 7, sigma_upper=5.0477717856495843e-09),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        obs = 'Kstar_BR'
        yrange = (0.5e-7, 7e-7)
        ylabel = r'$\langle \mathcal{B}(K^{\ast}) \rangle'

        s_min, s_max = 1, 6
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=dict(central=1.7e-7, sigma_upper=0.27e-7, sigma_lower=0.31e-7),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        s_min, s_max = 14.18, 16
        self.fig_pred(scen_obs=self.scenario_comparison(obs, 14, s_max),
                      measurement=dict(central=1.02e-7, sigma_upper=0.15e-7, sigma_lower=0.18e-7),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        s_min, s_max = 16, 19
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=dict(central=1.23e-7, sigma_upper=0.19e-7, sigma_lower=0.21e-7),
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        # B_d lifetime in GeV^-1
        tau_B = 1.519e-12 / 6.58212e-25

        obs = 'Kstar_J_1c_plus_J_2c'
        yrange = (-0.1e-8, 12e-8)  # tuple(np.array((-0.1e-20, 6e-20)) * inv_tau_B)
        ylabel = r'$\tau_B \, \times \, \langle J_{1c} + J_{2c} \rangle'

        s_min, s_max = 1.1, 6
        self.fig_pred(scen_obs=self.scenario_comparison(obs, 1, s_max),
                      measurement=None, rescale=tau_B,
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        s_min, s_max = 15, 19
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=None, rescale=tau_B,
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        yrange = (-0.05e-7, 5e-7)  # tuple(np.array((-0.05e-19, 2.5e-19)) * inv_tau_B)
        ylabel = r'$\tau_B \, \times \, \langle J_{1s} - 3 J_{2s} \rangle'
        obs = 'Kstar_J_1s_minus_3J_2s'
        s_min, s_max = 1.1, 6
        self.fig_pred(scen_obs=self.scenario_comparison(obs, 1, s_max),
                      measurement=None, rescale=tau_B,
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

        s_min, s_max = 15, 19
        self.fig_pred(scen_obs=self.scenario_comparison(obs, s_min, s_max),
                      measurement=None, rescale=tau_B,
                      ylabel=ylabel + '_{[%g,%g]}$' % (s_min, s_max), yrange=yrange)

    def all(self):
        import inspect

        for m in inspect.getmembers(self, predicate=inspect.ismethod):
            if m[0].startswith('fig'):
                m[1]()


if __name__ == '__main__':
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['font.size'] = 28
    matplotlib.rcParams['text.latex.preamble'] = r'''
    \usepackage[proportional]{libertine}
    \usepackage[libertine]{newtxmath}
    '''

    # enlarge ticks
    major = dict(size=8, width=1.4, pad=10)
    minor = dict(size=4, width=1.)
    matplotlib.rc('xtick.major', **major)
    matplotlib.rc('xtick.minor', **minor)
    matplotlib.rc('ytick.major', **major)
    matplotlib.rc('ytick.minor', **minor)
    matplotlib.rcParams['axes.linewidth'] = major['width']

    f = Spring2015()
    f.figSP()
    # f.figTT5()
    # f.fig_1()
    # f.all()
