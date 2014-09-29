#! /usr/bin/python
"""
Plot several distributions with their contours for paper
"""
from __future__ import print_function
import plotScript
from plotScript import ParameterDefinition
import plotUncertainty
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.patches
import matplotlib.pyplot as P
from matplotlib.lines import Line2D
import numpy as np
import os
import itertools
from collections import OrderedDict, defaultdict, namedtuple
from sets import Set
import sys

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
        """
        self.pars = ['Re{c7}', 'Re{c9}', 'Re{c10}']

        self.defs = OrderedDict()
        #(index, (min, max))
        self.defs['Re{c7}'] = ParameterDefinition(index=0, name='Re{c7}', min=-1, max=+1)
        self.defs['Re{c9}'] = ParameterDefinition(index=1, name='Re{c9}', min=-15, max=15)
        self.defs['Re{c10}'] = ParameterDefinition(index=2, name='Re{c10}', min=-15, max=15)
        """
        self.input_base = input_base

        self.scen = OrderedDict()
        """
        # remove some for testing and increasing speed
        if ignore_scenarios:
            for s in ignore_scenarios:
                del(self.scen[s])

        print("Operating on: %s" % str(self.scen.keys()))
        """
        # SM prediction for C7, C9, C10
        """
        self.sm_point = [-0.32741917, +4.27584794, -4.15077942]
        """
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
        cmd_template += ' --pmc-crop-outliers %d' % scenario.crop_outliers
        cmd_template += ' --2D-bins %d' % scenario.nbins
        if self.max_samples is not None:
            cmd_template += ' --select 0 %d' % self.max_samples
        cmd_template += ' --contours'
        if scenario.queue_output:
            cmd_template += ' --pmc-queue-output'

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
        fig.text(0.52, 0.04, tr.to_tex(def1.name), ha='center', va='center')
        fig.text(0.04, 0.52, tr.to_tex(def2.name), ha='center', va='center', rotation='vertical')

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

    def scI_all_vs_excl(self):
        """
        compare scI-all with scI-excl
        """

        scenarios = ('scI_excl', 'scI_all',)

        """
        # define bandwidths for each individual plot
        bw = 0.02
        self.comparison_bandwidths[(1, 0, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # top
        self.comparison_bandwidths[(0, 1, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # bottom
        self.comparison_bandwidths[(0, 0, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # bottom
        self.comparison_bandwidths[(1, 1, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # top
        self.comparison_bandwidths[(1, 0, 'Re{c9}', 'Re{c10}')] = bw * 1.5  # bottom
        self.comparison_bandwidths[(0, 1, 'Re{c9}', 'Re{c10}')] = bw * 1.4  # top
        """
        self.stack('Re{c7}', 'Re{c9}', scenarios=scenarios)
        P.savefig(self.out("%s_%s_0_1" % (scenarios)))

        self.stack('Re{c7}', 'Re{c10}', like_sign=True, scenarios=scenarios)
        P.savefig(self.out("%s_%s_0_2" % (scenarios)))

        self.stack('Re{c9}', 'Re{c10}', flip_order=True, scenarios=scenarios)
        P.savefig(self.out("%s_%s_1_2" % (scenarios)))

    def one_dim_nuisance_KMPW(self):
        """
        Plot prior and posterior 1D of KMPW form factors for different subscenarios in one plot.
        """
        """
        plotScript.py sc1_BPll_parameter_samples_12.hdf5_merge  --nuisance --pmc-crop-outliers 200 --single-1D 12 --use-KDE --bandwidth 0.004
        plotScript.py sc1_BPll_parameter_samples_12.hdf5_merge  --nuisance --pmc-crop-outliers 200 --single-1D 13 --use-KDE --bandwidth 0.04
        """

        scenarios = ('scI_all',)

        class ParameterProperty(object):
            "Hold properties for each parameter plot"
            def __init__(self, p, bandwidth=None, legend_pos='upper right'):
                self.index = p
                # None triggers automatic determination of bandwidth
                self.bandwidth = bandwidth
                self.legend_pos = legend_pos

#        parameters = ("B->K::F^p(0)@KMPW2010", "B->K::b^p_1@KMPW2010")
        props = {  "B->K^*::F^V(0)@KMPW2010": ParameterProperty(10),
                   "B->K^*::b^V_1@KMPW2010": ParameterProperty(11),
                   "B->K^*::F^A1(0)@KMPW2010": ParameterProperty(14),
                   "B->K^*::b^A1_1@KMPW2010": ParameterProperty(15),
                   "B->K^*::F^A2(0)@KMPW2010": ParameterProperty(16),
                   "B->K^*::b^A2_1@KMPW2010": ParameterProperty(17),
                   "B->K::F^p(0)@KMPW2010": ParameterProperty(25),
                   "B->K::b^p_1@KMPW2010": ParameterProperty(26, legend_pos='upper left')
                }

        marginal_styles = {'scI_all': dict(color=self.scen['scI_all'].c, linestyle='solid')}

        line_width = 1.5
        for m in marginal_styles.values():
            m['linewidth'] = line_width

        prior_style = dict(color='black', linestyle='dotted', linewidth=line_width)

        x_size = 8; y_size = 6

        for k, p in props.iteritems():

            for s in scenarios:

                fig = P.figure(figsize=(x_size, y_size))
                ax = fig.add_subplot(111)

                self.margs[s].use_histogram = False
                self.margs[s].kde_bandwidth = p.bandwidth

                # don't skip this plot
                self.margs[s].use_nuisance = True
                self.margs[s].use_contours = False
                self.margs[s].plot_prior = True

                self.margs[s].one_dimensional(p.index, marginal_style=marginal_styles[s], prior_style=prior_style,
                                              legend_label='$\mathrm{posterior}$', prior_label='$\mathrm{prior}$')

                P.legend(loc=p.legend_pos)
                P.xlabel(self.margs[s].tr.to_tex(k))
                ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    #            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                P.setp(ax.get_yticklabels(), visible=False)
                P.tight_layout()
                P.savefig(self.out('nuis_1D_FF_%s' % k))

    def subleading_together(self, scenario, indices, bandwidths):
        """
        Plot prior and posterior 1D for three subleading parameters with common prior.

        :param scenario:
            string

        :param indices:
            sequence of parameter indices defining parameter in file

        :param bandwidths:
            KDE bandwidth, one for each parameter
        """

        marg = self.margs[scenario]

        marg.use_nuisance = True
        marg.use_histogram = False
        marg.use_contours = False

        temp_font_size = matplotlib.rcParams['font.size']
        matplotlib.rcParams['font.size'] = 16

        class Prop(object):
            """properties of each figure"""
            def __init__(self, name, legend_label="",
                         style=dict(color=self.scen[scenario].c, linestyle='solid')):
                self.name = name
                self.legend_label = legend_label
                self.style = style

        props = (Prop(12, 0.005), Prop(13, 0.015))

        prior_style=dict(color='black', linestyle='dashed', dashes=[4, 3])


        props = (Prop("B->K^*ll::A_perp^L_uncertainty@LargeRecoil",
                      legend_label=r"$\chi=\perp$",
                      style=dict(color='blue', marker='', linestyle='dashed')),
                 Prop("B->K^*ll::A_par^L_uncertainty@LargeRecoil",
                      legend_label=r"$\chi=\,\parallel$",
                      style=dict(color='OrangeRed', marker='', linestyle='-.')),
                 Prop("B->K^*ll::A_0^L_uncertainty@LargeRecoil",
                      legend_label=r"$\chi=0$",
                      style=dict(color='green', linestyle='solid')))

        line_width = 1.5
        for p in props:
            p.style['linewidth'] = line_width

        prior_style = dict(color='grey', alpha=0.3, linewidth=line_width, filled=True)
        prior_label = r"$\mathrm{prior}$"

        for i,p in enumerate(props):
            if bandwidths is not None:
                marg.use_histogram = False
                marg.kde_bandwidth = bandwidths[i]
            else:
                marg.use_histogram = True

            # don't skip this plot
            marg.plot_prior = (i == 0)

            marg.one_dimensional(indices[i], marginal_style=p.style, prior_style=prior_style,
                                 legend_label=p.legend_label, prior_label=prior_label)

        P.xlabel(r"$\zeta_{K^{\ast}}^{L\chi}$")
        ax = P.gca()
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        P.setp(ax.get_yticklabels(), visible=False)

        # sort entries by label to push prior at the end
        handles, labels = ax.get_legend_handles_labels()
        import operator
        hl = sorted(zip(handles, labels), key=operator.itemgetter(1), reverse=False)
        handles2, labels2 = zip(*hl)
        ax.legend(handles2, labels2)
        P.legend(handles2, labels2, loc='upper right')
        P.tight_layout()
        P.savefig(self.out(scenario + '_subleading'))

    def subleading_separate(self):
        """Plot subleading_separate 1D marginals separately with 1- and 2-sigma regions"""

        scenario = 'scI_posthep13'
        marg = self.margs[scenario]

        marg.use_nuisance = True
        marg.use_histogram = False
        marg.use_contours = True

        class Prop(object):
            """properties of each figure"""
            def __init__(self, i, bandwidth=None):
                # index of parameter in hdf5
                self.i = i
                # kde bandwidth
                self.bandwidth = bandwidth


        temp_font_size = matplotlib.rcParams['font.size']
        matplotlib.rcParams['font.size'] = 16

        props = (Prop(12, 0.005), Prop(13, 0.015))

        prior_style=dict(color='black', linestyle='dashed', dashes=[4, 3])
        one_sigma_style = dict(facecolor='blue', alpha=0.7)
        two_sigma_style = dict(facecolor='blue', alpha=0.4)

        # patches for the legend
        rect_one = matplotlib.patches.Rectangle((0,0), 1, 1, **one_sigma_style)
        rect_two = matplotlib.patches.Rectangle((0,0), 1, 1, **two_sigma_style)
        rect_prior = matplotlib.lines.Line2D((0,0), (1, 1), **prior_style)

        for p in props:
            P.figure(figsize=(6,5))
            marg.kde_bandwidth = p.bandwidth
            marg.one_dimensional(p.i, prior_label=r"$\mathrm{prior}$", prior_style=prior_style,
                                 one_sigma_style=one_sigma_style, two_sigma_style=two_sigma_style)

            P.setp(P.gca().get_yticklabels(), visible=False)
            if p.i == 12:
                P.legend([rect_one, rect_two, rect_prior], [r'$1 \sigma$', r'$2 \sigma$', r'$\mathrm{prior}$'])
            P.tight_layout()
            P.savefig(self.out('%s_%d' % (scenario, p.i)))

        # restore
        matplotlib.rcParams['font.size'] = temp_font_size

    def single_scenario(self):
        """
        Plot one scenario by itself at a time.
        """

        square_figure()

        # loop over scenarios
        for k, name in enumerate(self.scen.keys()):
            sc = self.scen[name]

            for i, d1 in enumerate(sc.defs):
                for j, d2 in enumerate(sc.defs[i + 1:]):
                    P.clf()
                    self.single_panel(d1, d2, SM_point=True, local_mode=True, scenarios=(name,), desired_levels=(0.683, 0.954, 0.9973))
                    P.savefig(self.out('%s_%d_%d' % (name, i, j)))

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

    def credibility_regions(self):
        """
        Extract marginal 1D credibility intervals.
        Results are printed to stdout
        """

        """
        plotScript.py pmc_final_scI_all.hdf5 --single-1D 0 --contours --1D-bins 200 --use-data-range 0.05 --pmc-queue-output --pmc-crop 50

        plotScript.py pmc_final_scI_all.hdf5 --single-1D 1 --contours --1D-bins 150 --use-data-range 0.05 --pmc-queue-output --pmc-crop 50
        plotScript.py pmc_final_scI_all.hdf5 --single-1D 1 --contours --1D-bins 150 --use-data-range 0.05 --pmc-queue-output --pmc-crop 50 --use-KDE --bandwidth 0.08

        plotScript.py pmc_final_scI_all.hdf5 --single-1D 2 --contours --1D-bins 150 --use-data-range 0.05 --pmc-queue-output --pmc-crop 50
        plotScript.py pmc_final_scI_all.hdf5 --single-1D 2 --contours --1D-bins 150 --use-data-range 0.05 --pmc-queue-output --pmc-crop 50 --use-KDE --bandwidth 0.05

        same for wide priors for c7. For 9 use bandwidth 0.018

        plotScript.py sc1_all_wide.hdf5 --single-1D 2 --contours --1D-bins 220 --use-data-range 0.05
        plotScript.py sc1_all_wide.hdf5 --single-1D 2 --contours --1D-bins 300 --use-data-range 0.05 --use-KDE --bandwidth 0.030
        """

        # common range for all_nuis and all_wide
        tight_defs = OrderedDict()
#        tight_defs['Re{c7}'] = plotScript.ParameterDefinition(name='Re{c7}', min=-0.8, max=+0.8, index=self.pars.index('Re{c7}'))
        tight_defs['Re{c9}'] = plotScript.ParameterDefinition(name='Re{c9}', min=-9, max=+8.5, index=self.pars.index('Re{c9}'))
#        tight_defs['Re{c10}'] = plotScript.ParameterDefinition(name='Re{c10}', min=-6, max=+6, index=self.pars.index('Re{c10}'))

        # separate n_bins (scenario, parameter, use_histogram)
        nbins_default = 500
        nbins = defaultdict(lambda : nbins_default)
        nbins[('all_nuis', 'Re{c7}', True)] = 220
        nbins[('all_nuis', 'Re{c9}', True)] = 220
        nbins[('all_nuis', 'Re{c10}', True)] = 220
        nbins[('all_wide', 'Re{c7}', True)] = 180
        nbins[('all_wide', 'Re{c9}', True)] = 180
        nbins[('all_wide', 'Re{c10}', True)] = 180

        # bandwidths
        bandwidths = defaultdict(lambda : 0.01)
        bandwidths[('all_nuis', 'Re{c7}')] = 0.005
        bandwidths[('all_nuis', 'Re{c9}')] = 0.09
        bandwidths[('all_nuis', 'Re{c10}')] = 0.02
        bandwidths[('all_wide', 'Re{c7}')] = 0.01
        bandwidths[('all_wide', 'Re{c9}')] = 0.06
        bandwidths[('all_wide', 'Re{c10}')] = 0.06

        for s in ('all_nuis',):
            marg = self.margs[s]
            marg.fixed_1D_binning = True

            for d in tight_defs.values():
                for histo in (True, False):

                    marg.cuts[d.i] = d.range

                    marg.use_histogram = histo
                    marg.kde_bandwidth = bandwidths[(s, d.name)]

                    marg.one_dim_n_bins = nbins[(s, d.name, histo)]
                    marg.one_dimensional(d.i)

                    P.savefig(self.out('%s_one_dim_%d_%d' % (s, d.i, histo)))
                    P.clf()

class ObservableProperties(object):

    def __init__(self, name, index, range=None, n_bins=150, mode_precision=3, plot=False,
                 histo_style=None, credibility_style=None, n_ticks=None, y_labels=False):
        self.name = name
        self.index = index
        self.range = range
        self.n_bins = n_bins
        self.mode_precision = mode_precision
        self.plot = plot
        self.histo_style = histo_style
        self.credibility_style = credibility_style
        self.n_ticks = n_ticks
        self.y_labels = y_labels

def pull(input_base, output_base, ext='.pdf', mode=0, SM=False):
    """
    Use input file /.th/pcl128c/scratch/beaujean/EOS/2012-04-12/Scenario1_all_nuis/sc1_all_nuis_gof_0.hdf5
    with global mode (SM like). Old modes before bugfixes in 2012-03-19
    """
    import plotPull

    output_dir = os.path.join(output_base, "pull")

    input_file = os.path.join(input_base, "sc1_all_nuis_gof_%d.hdf5" % mode)
    print(input_file)
    if SM:
#        input_file = "/.th/pcl128c/scratch/beaujean/EOS/2012-04-23/Scenario1_SM_all/sc1_SM_all_gof_0.hdf5"
        raise NotImplementedError("SM prediction missing")

    def out(name):
        """ Create output file name"""
        return os.path.join(output_dir, ('SM_' if SM else '') + name) + ext

    cmd_line = input_file.split()
    pull = plotPull.factory(cmd_line)

    x_size = 8
    y_size = 6.5

    ###
    # two plots next to each other
    ###
    adjust = dict(sig_max=3, padding=0.1, n_col=2, x_size=x_size, y_size=1.8 * y_size, left_adjust=0.195)
    obs = ['BR', 'F_L']
    obs = [plotPull.make_constraint_names(o) for o in obs]
    obs = list(itertools.chain.from_iterable(obs))
    pull.plot(obs, top_adjust=0.85, **adjust)
    P.savefig(out("pull_K*_BR_F_L"))

    obs = ['A_FB', 'A_T_2', 'S_3']
    obs = [plotPull.make_constraint_names(o) for o in obs]
    # create flattened list, no more list of lists
    obs = list(itertools.chain.from_iterable(obs))
    pull.plot(obs, top_adjust=0.955, legend=False, **adjust)
    P.savefig(out("pull_K*_A_FB_A_T_2_S_3"))

    pull.plot(['B^0_s->mu^+mu^-::BR_limit'])
    P.savefig(out("pull_B_S_BR_limit"))

    ###
    # two plots next to each other
    ###
    adjust = dict(sig_max=3, padding=0.1, n_col=3, x_size=x_size, y_size=0.8 * y_size,
              left_adjust=0.19, top_adjust=0.86, bottom_adjust=0.15, handle_length=1.5)

    pull.plot(plotPull.make_constraint_names('BR', decay='B^+->K^+mu^+mu^-'), **adjust)
    P.savefig(out("pull_K_BR"))

    obs = ['B^0->K^*0gamma::BR', 'B^0->K^*0gamma::S_K+C_K']
    pull.plot(obs, **adjust)
#              sig_max=2, padding=0.1, n_col=3, x_size=x_size, y_size=0.8 * y_size,)
#              left_adjust=0.12, top_adjust=0.86, bottom_adjust=0.15)
    P.savefig(out("pull_gamma"))

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

class Fall2013(object):

    # set up object
    input_base, output_base = input_output()
    max_samples = None
    fig_size = 6 # inches, same for x and y

    def figI(self):
        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        scenarios = ('sm_posthep13', 'scIII_posthep13')#, 'scIII_posthep13hpqcd')
        for name in scenarios:
            marg.scen[name] = Scenario(os.path.join(marg.input_base, 'pmc_' + name + '.hdf5'), 'OrangeRed',
                                              crop_outliers=200, nbins=150)
        marg.read_data()

        wide_figure(self.fig_size)
        marg.subleading_together(scenario='sm_posthep13', indices=(9, 20, 18), bandwidths=(0.005, 0.01, 0.015))
        wide_figure(self.fig_size)
        marg.subleading_together(scenario='scIII_posthep13', indices=(15, 26, 24), bandwidths=(0.01, 0.02, 0.03))
#         marg.subleading_together(scenario='scIII_posthep13hpqcd', indices=(15, 26, 24), bandwidths=(0.01, 0.02, 0.03))

    def figII(self):

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        ###
        # scenarios
        ###

        marg.scen['scI_posthep13'] = Scenario(os.path.join(marg.input_base, 'pmc_scI_posthep13.hdf5'), 'OrangeRed',
                                        crop_outliers=200, nbins=300,
                                        local_mode=[[-0.348441713,   3.787226592, -4.420530192],
                                                    [ 0.5021320352, -4.568457245,  4.25129282]])
    #     marg.subleading_together()

        marg.scen['scI_quim1'] = Scenario(os.path.join(marg.input_base, 'pmc_scI_quim1.hdf5'), 'Blue',
                                          crop_outliers=200, nbins=300,
                                          local_mode=[[-0.3323645523, 2.445929253, -4.07158241],
                                                      [ 100, 100, 100] # outside of plot region => no mode
                                                      ])
#         marg.scen['scI_all_nuis'] = Scenario(os.path.join(marg.input_base, 'pmc_scI_all_nuis.hdf5'), 'Grey',
#                                              crop_outliers=200,
#                                              local_mode=[[-0.294991910838,    3.731820480717,  -4.140554057902],
#                                                          [ 0.41787049285,  -4.639111764728,  3.994616452063]])

        marg.read_data()

        ###
        # settings for comparing scenarios
        ###
        name = 'Re{c7}'
        nticks = 5
        par_def0 = ParameterDefinition(name=name, min=-0.5, max=-0.1, index=0)
        par_def0.major_locator = ticker.FixedLocator(np.linspace(par_def0.min, par_def0.max, nticks))
        par_def1 = ParameterDefinition(name=name, min=0.3, max=0.7, index=0)
        par_def1.major_locator = ticker.FixedLocator(np.linspace(par_def1.min, par_def1.max, nticks))

        # 0 = SM, 1 = flipped sign
        marg.comparison_defs[('Re{c7}' , 0)] = par_def0
        marg.comparison_defs[('Re{c7}' , 1)] = par_def1
        marg.comparison_defs[('Re{c9}' , 0)] = ParameterDefinition(name='Re{c9}', min=-0.5, max=+5.5, index=1)
        marg.comparison_defs[('Re{c9}' , 1)] = ParameterDefinition(name='Re{c9}', min=-8, max=-2, index=1)
        marg.comparison_defs[('Re{c10}', 0)] = ParameterDefinition(name='Re{c10}', min=-7, max=-1, index=2)
        marg.comparison_defs[('Re{c10}', 1)] = ParameterDefinition(name='Re{c10}', min=+1, max=+7, index=2)

        ###
        # Create plot for large range to properly weight both modes, then zoom in
        ###
        large_def = (ParameterDefinition(name='Re{c7}', min=-0.5, max=+0.6, index=0),
                     ParameterDefinition(name='Re{c9}', min=-7, max=+6, index=1),
                     ParameterDefinition(name='Re{c10}', min=-6, max=+6, index=2),
                     )

        # bandwidths for each plane
        marg.scen['scI_posthep13'].set_bandwidth(0, 1, 0.01)
        marg.scen['scI_posthep13'].set_bandwidth(0, 2, 0.008)
        marg.scen['scI_posthep13'].set_bandwidth(1, 2, 0.008)

        marg.scen['scI_quim1'].set_bandwidth(0, 1, 0.01)
        marg.scen['scI_quim1'].set_bandwidth(0, 2, 0.008)
        marg.scen['scI_quim1'].set_bandwidth(1, 2, 0.01)

        for scenario in ('scI_posthep13', 'scI_quim1'):
            for i in range(3):
                for j in range(i + 1, 3):
                    marg.compute_marginal(large_def[i], large_def[j], scenario, solution=(0, 1))

        marg.single_scenario()
        combinations = (('scI_quim1',), ('scI_posthep13',), ('scI_posthep13', 'scI_quim1'),
                        )
    #                     ('scI_all_nuis',),
    #                     ('scI_all_nuis', 'scI_posthep13'), ('scI_all_nuis', 'scI_quim1'))

        marg.compare_scenarios(combinations, desired_levels=(0.683, 0.954, 0.9973), local_mode=[True, True])

#     marg.subleading_separate()

    def figIII(self):
        """Scenario III marginals for C_i vs C'_i"""

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        defs = (ParameterDefinition(index=0, name='Re{c7}', min=-0.5, max=0.7),
                ParameterDefinition(index=1, name='Re{c9}', min=-6.5, max=5.5),
                ParameterDefinition(index=2, name='Re{c10}', min=-6, max=6))

        scenarios = ['scIII_posthep13']
        s = Scenario(os.path.join(self.input_base, 'pmc_' + scenarios[-1] + '.hdf5'), 'OrangeRed', bandwidth_default=0.005,
                                  crop_outliers=200, nbins=300,
                                  local_mode=[[-0.3294454156375440, +3.7033084792658708, -4.8352440600734292,
                                               -0.0705098937666282, -1.0515469382939286, -0.5277211018937606],
                                              [+0.4891045191541478, -4.2679436027176978, +4.4476234333920459,
                                               +0.1114889290368637, +0.5703468115735241, +0.8397671055132904],
                                              [-0.0157734421894522, -1.1321289138990118, +0.3796536153807689,
                                               -0.4248361956296707, +4.2441573979695137, +4.4729213662276806],
                                              [+0.1508212580097578, +0.1897372098757005, -0.6023744149054687,
                                               +0.4237873414046214, -3.9155364861377104, -4.4686368052186660]])
        s.set_bandwidth(0, 3, 0.01)
        s.set_bandwidth(1, 4, 0.025)
        s.set_bandwidth(2, 5, 0.013)

        marg.scen[scenarios[-1]] = s

        scenarios.append('scIII_posthep13hpqcd')
        s = Scenario(os.path.join(self.input_base, 'pmc_' + scenarios[-1] + '.hdf5'), 'Blue', bandwidth_default=0.005,
                                  crop_outliers=200, nbins=300,
                                  local_mode=[[-0.3478138434026574, +3.6192275849809392, -4.1450675009769320,
                                               -0.0210563264757002, +1.0799571832442161, +0.1809625656629924],
                                              [+0.4869893356704660, -4.4906101273214931, +4.4355261568265361,
                                               +0.1329187879824148, -0.3929626383549490, +0.3604274141650922],
                                              [-0.0230814743653742, -0.5250663856254302, +0.5016281568890346,
                                               -0.4129907710336000, +3.9653288279366410, +4.3054259043756611],
                                              [+0.1511251755255345, -0.4032028484462637, -0.4852016080815704,
                                               +0.4059691733431722, -3.8706811714180409, -4.2834154791241801]])
        s.set_bandwidth(0, 3, 0.01)
        s.set_bandwidth(1, 4, 0.018)
        s.set_bandwidth(2, 5, 0.013)

        marg.scen[scenarios[-1]] = s

        marg.read_data()

        # predictions in the SM

        primed_defs = [ParameterDefinition(index=3, name="Re{c7'}" , min=-0.6, max=0.6),
                       ParameterDefinition(index=4, name="Re{c9'}" , min=-6, max=6),
                       ParameterDefinition(index=5, name="Re{c10'}", min=-6, max=6),]
        primed_defs[0].major_locator = ticker.FixedLocator(np.linspace(primed_defs[0].min, primed_defs[0].max, 7))

#         for d in primed_defs:
#             m.cuts[d.i] = d.range
        primed_predictions = (0, 0, 0)

        # place labels identifying the solutions
        mode_labels = (('A', ((-0.26, -0.05), (1.2, 0), (-3.5, -1.2))),
                       ('B', ((0.34, 0.02),  (-3.2, 0), (2.5, 0.5))),
                       ('C', ((0.01, -0.32), (-1, 2), (0.5, 2.7))),
                       ('D', ((0.1, 0.25), (0, -2.8), (-1, -3.5))))

        square_figure(self.fig_size)

        for i in range(3):
            marg.single_panel(defs[i], primed_defs[i], scenarios=scenarios,
                              desired_levels=(0.683, 0.954), local_mode=(True, True))

            for p, (lab, loc) in zip(s.local_mode, mode_labels):
                P.text(loc[i][0], loc[i][1], '$'+lab+'^{\prime}$')

            adjust_subplot()
            P.savefig(marg.out('scIII_%d' % i))

    def figIV(self):
        '''scII plot'''

        marg = MarginalContours(self.input_base, self.output_base, max_samples=self.max_samples)

        ###
        # parameter definitions
        ###
        def9      = ParameterDefinition(name='Re{c9}',  min=1, max=6, index=0)
        def9prime = ParameterDefinition(name="Re{c9'}", min=-2, max=3, index=1)

        sc_names = ('scII_posthep13', 'scII_posthep13hpqcd')
        local_modes = ([[3.601411695, 0.5116544958]], [[3.761138544, 0.3787229407]])
        marg.scen[sc_names[0]] = Scenario(os.path.join(self.input_base, 'pmc_%s.hdf5' % sc_names[0]), 'OrangeRed',
                                          bandwidth_default=0.025,
                                          queue_output=False, crop_outliers=200, local_mode=local_modes[0], defs=[def9, def9prime])
        marg.scen[sc_names[1]] = Scenario(os.path.join(self.input_base, 'pmc_%s.hdf5' % sc_names[1]), 'Blue',
                                          bandwidth_default=0.017,
                                          queue_output=False, crop_outliers=200, local_mode=local_modes[1], defs=[def9, def9prime])

        marg.read_data()

        square_figure(self.fig_size)
        # 1 and 2 sigma contours
        marg.single_panel(def9, def9prime, SM_point=True, local_mode=[True, True], scenarios=sc_names,
                          desired_levels=(0.683, 0.954), label=True)
        adjust_subplot()
        P.savefig(marg.out('scII'))

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

    f = Fall2013()
    f.figII()
#    f.all()
