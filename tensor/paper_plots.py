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

def wide_figure(x_size=8, ratio=4/3.0, left=0.08, right=0.95, top=0.95, bottom=0.15):
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
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15)

class Scenario(object):
    def __init__(self, file, color, nbins=200, alpha=0.4, sigma='2 sigma',
                 bandwidth_default=None, two_sigma_color=None, queue_output=True,
                 crop_outliers=200, local_mode=None, defs={}):
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

    def get_bandwidth(self, par1, par2):
        try:
            return self.__bandwidth[(par1,par2)]
        except KeyError:
            return self.__bandwidth_default

    def set_bandwidth(self, par1, par2, value):
        self.__bandwidth[(par1, par2)] = value

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

        self.output_base = os.path.join(output_base, "contour")
        if not os.path.isdir(self.output_base):
            os.mkdir(self.output_base)
        self.ext = ext

        self.input_base = input_base

        self.scen = OrderedDict()
        # SM prediction for C7, C9, C10
        self.sm_point = {'Re{c7}':-0.33726473, 'Re{c9}':+4.27342842, 'Re{c10}':-4.16611761}
        self.sm_point_style = dict(marker = 'D', markersize = 8, color = 'black')

        best_fit_point_style = dict(self.sm_point_style)
        best_fit_point_style['marker'] = 'x'
        best_fit_point_style['markeredgewidth'] = 3.0

        next_best_fit_point_style = dict(best_fit_point_style)
        next_best_fit_point_style['marker'] = '+'
        next_best_fit_point_style['color'] = 'Blue'

        self.best_fit_points_style = [[best_fit_point_style]*4, [next_best_fit_point_style]*4]

        # highest to lowest contour
        self.contour_styles = [dict(linestyle='solid'), dict(linestyle='dashed'), dict(linestyle='dashdot')]

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
#         cmd_template += ' --pmc-crop-outliers %d' % scenario.crop_outliers
        cmd_template += ' --2D-bins %d' % scenario.nbins
        if self.max_samples is not None:
            cmd_template += ' --select 0 %d' % self.max_samples
        cmd_template += ' --contours'
        cmd_template += ' --mcmc --pypmc --skip-init 0.2'

        for p in scenario.defs:
            cmd_template += ' --cut %d %s %s' % (p.i, p.min, p.max)

        return cmd_template.split()

    def read_data(self):

        self.margs = {}
        for k in self.scen.keys():
            self.margs[k] = plotScript.factory(self.command_template(self.scen[k]))

    def compute_marginal(self, def1, def2, scenario, solution=0):
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
                     desired_levels=None, label=True):
        """
        Single marginal plot to compare two scenarios

        First scenario is plotted with filled colored regions,
        the second is plotted with contour lines

        :param solution:
            Index to identify a region of zoom (a *solution*) into the 2D marginal.
        """

        assert(len(scenarios) in (1, 2))

        # parameter indices in EOS output file
        i = def1.i
        j = def2.i

        # compute and cache densities
        for s in scenarios:
            self.compute_marginal(def1, def2, s, solution)

        for k, s in enumerate(scenarios):
            # retrieve original range used to create prob_density
            # then zoom will work correctly
            density, xrange, yrange = self.__density_cache[(i, j, s, solution)]
            line = bool(k)
            artist = self.margs[s].contours_two(xrange, yrange, density,
                                                color=self.scen[s].c, line=line, grid=True,
                                                desired_levels=desired_levels)
            if line:
                for n, l in enumerate(desired_levels):
                    P.setp(artist.collections[n], **self.contour_styles[-n-1])

        if SM_point:
            P.plot(self.sm_point.get(def1.name, 0.), self.sm_point.get(def2.name, 0.), **self.sm_point_style)
        if any(local_mode):
            for n, s in enumerate(scenarios):
                if local_mode[n]:
                    for style, p in zip(self.best_fit_points_style[n], self.scen[s].local_mode):
                        P.plot(p[i], p[j], **style)

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
            P.subplot(2,1,i)
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
        P.figure(figsize=(8,8))

        for i, p1 in enumerate(self.pars):
            for j, p2 in enumerate(self.pars[self.pars.index(p1) + 1:]):
                P.clf()
                # draw SM point
                P.plot(self.sm_point[self.defs[p1].i], self.sm_point[self.defs[p2].i], **self.sm_point_style)

                for k in self.scen.keys():
                    P.xlabel(self.margs[k].tr.to_tex(p1))
                    P.ylabel(self.margs[k].tr.to_tex(p2))
                    # draw contours
                    CS = self.margs[k].contours_two(self.defs[p1].range, self.defs[p2].range, self.scen[k].prob[(p1,p2)],
                                   desired_levels=self.scen[k].sigma, color=self.scen[k].c, grid=False)
                    P.setp(CS.collections[1], alpha=self.scen[k].alpha)
                    c = self.scen[k].two_sigma_color
                    if c is not None:
                        P.setp(CS.collections[1], color=c)

                    # don't whiten beyond 2 sigma, so ignore lowest contour fill
                    #P.setp(CS.collections[0], alpha=0.0)
                    CS.collections[0].remove()

                    P.savefig(self.out('overlay_%d_%d_%s' % (i, self.pars.index(p1) + j + 1, k)))

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
    data = data[:, i:j+1:j-i]
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
    fig_size = 6 # inches, same for x and y

    def figSP(self):

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        ###
        # scenarios
        ###
        s = 'scSP_Bsmumu'
        scenarios = [s]

        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_scSP_Bsmumu.hdf5'), 'OrangeRed',
                                nbins=300)
#                                         local_mode=[[-0.348441713,   3.787226592, -4.420530192],
#                                                     [ 0.5021320352, -4.568457245,  4.25129282]])

        marg.read_data()
        m = marg.margs[s]
        data = m.out.samples

        ###
        # transformations
        ###
        transform(data, 0, 2)
        transform(data, 4, 6)

        defs = [ParameterDefinition(index=0, name=r"$\Re(\mathcal{C}_S + \mathcal{C}_S^{\prime})$", min=-1, max=1),
                ParameterDefinition(index=2, name=r"$\Re(\mathcal{C}_S - \mathcal{C}_S^{\prime})$", min=-1, max=1),
                ParameterDefinition(index=4, name=r"$\Re(\mathcal{C}_P + \mathcal{C}_P^{\prime})$", min=-1, max=1),
                ParameterDefinition(index=6, name=r"$\Re(\mathcal{C}_P - \mathcal{C}_P^{\prime})$", min=-1, max=1),
                ]

        marg.scen[s].set_bandwidth(0, 2, 0.015)
        marg.scen[s].set_bandwidth(2, 6, 0.008)

        square_figure(self.fig_size)

        for ii in [(0,1), (1,3)]:
            marg.single_panel(defs[ii[0]], defs[ii[1]], scenarios=scenarios,
                              desired_levels=(0.683, 0.954), local_mode=(False, False))

            adjust_subplot()
            P.savefig(marg.out(s + str(ii)))

    def figTT5(self):

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        ###
        # scenarios
        ###
        s = 'scTT5_FH'
        scenarios = [s]

        marg.scen[s] = Scenario(os.path.join(marg.input_base, 'mcmc_scTT5_FH.hdf5'), 'OrangeRed',
                                nbins=300)
#                                         local_mode=[[-0.348441713,   3.787226592, -4.420530192],
#                                                     [ 0.5021320352, -4.568457245,  4.25129282]])

        marg.read_data()
        m = marg.margs[s]
        data = m.out.samples

        ###
        # transformations
        ###
        for ii, name in [((0,1), 'CT'), ((2,3), 'CT5')]:
            data[:, -1] = betrag(data, ii[0], ii[1])
            m.out.par_defs[-1].nuisance = False
            m.out.par_defs[-1].min = 0.0
            m.out.par_defs[-1].max = 1.45
            m.use_contours = True
            m.fixed_1D_binning = False
            m.nBins[-1] = 200
            
            P.clf()
            m.one_dimensional(data.shape[1]-1)
            P.savefig(marg.out(s+'_Betrag_' + name))

    def all(self):
        import inspect

        for m in inspect.getmembers(self, predicate=inspect.ismethod):
            if m[0].startswith('fig'):
                m[1]()

if __name__ == '__main__':
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['font.size'] = 22

    # enlarge ticks
    major = dict(size=8, width=1.4, pad=10)
    minor = dict(size=4, width=1.)
    matplotlib.rc('xtick.major', **major)
    matplotlib.rc('xtick.minor', **minor)
    matplotlib.rc('ytick.major', **major)
    matplotlib.rc('ytick.minor', **minor)
    matplotlib.rcParams['axes.linewidth'] = major['width']

    f = Spring2015()
#     f.figSP()
    f.figTT5()
#    f.all()
