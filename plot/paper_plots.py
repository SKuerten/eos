#! /usr/bin/python
"""
Plot several distributions with their contours for paper
"""

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
        self.sm_point = {'Re{c7}':-0.32741917, 'Re{c9}':+4.27584794, 'Re{c10}':-4.15077942}
        self.sm_point_style = dict(marker = 'D', markersize = 8, color = 'black')

        best_fit_point_style = dict(self.sm_point_style)
        best_fit_point_style['marker'] = 'x'
        best_fit_point_style['markeredgewidth'] = 3.0

        self.best_fit_points_style = [best_fit_point_style]*2

        self.max_samples = max_samples
        self.read_data()

        # store expensive calculation of contours
        self.__density_cache = {}

        # ranges and bandwidths for scenario comparison
        self.scen_comp = {}

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

#         for p in self.defs.itervalues():
#             cmd_template += ' --cut %d %s %s' % (p.i, p.min, p.max)

        return cmd_template.split()

    def read_data(self):

        self.margs = {}
        for k in self.scen.keys():
            self.margs[k] = plotScript.factory(self.command_template(self.scen[k]))

    def single_panel(self, pos1, pos2, def1, def2, SM_point=True, local_mode=True, scenarios=('all_wide', 'all_nuis')):
        """
        Single marginal plot to compare two scenarios

        First scenario is plotted with filled colored regions,
        the second is plotted with contour lines
        """

        assert(len(scenarios) in (1, 2))

        # parameter indices in EOS output file
        i = def1.i
        j = def2.i
#         i = self.pars.index(def1.name)
#         j = self.pars.index(def2.name)

        # compute and cache densities
        for s in scenarios:
            if not self.__density_cache.has_key((i, j, s)):
                bandwidth = self.scen_comp.comparison_bandwidths.get((s, pos1, pos2, def1.name, def2.name), None)
                if bandwidth is not None:
                    self.margs[s].use_histogram = False
                    self.margs[s].kde_bandwidth = bandwidth
                    print("bandwidth = %g " % bandwidth)
                else:
                    self.margs[s].use_histogram = True
                density = self.margs[s].two_dimensional(i, j)
                self.__density_cache[(i, j, s)] = density
                print("denstity sum %g" % density.sum())
        P.cla()

        for k, s in enumerate(scenarios):
            # retrieve original range used to create prob_density
            # then zoom will work correctly
            xrange = (def1.min, def2.max)
            yrange = (def2.min, def2.max)
            artist = self.margs[s].contours_two(xrange, yrange, self.__density_cache[(i, j, s)],
                                                color=self.scen[s].c, line=bool(k), grid=True)

        if SM_point:
            P.plot(self.sm_point[i], self.sm_point[j], **self.sm_point_style)
        if local_mode:
            for style, p in zip(self.best_fit_points_style, self.scen[scenarios[0]].local_mode):
                P.plot(p[i], p[j], **style)

        ax = P.gca()

        # some doesn't work in contours_two
        # so turn on manually
        ax.grid()

        if hasattr(def1, 'major_locator'):
            ax.xaxis.set_major_locator(def1.major_locator)
        ax.set_xlim(def1.range)

        if hasattr(def2, 'major_locator'):
            ax.yaxis.set_major_locator(def2.major_locator)
        ax.set_ylim(def2.range)
#         ax.set_aspect('auto', adjustable='box')
#         ax.set_aspect('equal')
#         P.axis('equal')

    def all_nuis_stack(self, def1, def2, like_sign=False, flip_order=False, scenarios=('all_wide', 'all_nuis')):
        """
        Plot two panels, one on top of the other with
        marginal distribution for two parameters.

        Keyword arguments:
        def1, def2 -- ParameterDefinition of both parameters.
        like_sign -- If true, use +,+ and -,-. Default: false, then use -+, +-.
        """

        # choose x_size such that fonts are readible
        # y size manually adjusted so actual plot frame (not the figure!) has 1:1 aspect ratio
        x_size = 5
        fig = P.figure(figsize=(x_size, 1.8 * x_size))

        # use -+ or +-?
        ind = (1, 0)
        if like_sign:
            ind = (1, 1)
        if flip_order:
            ind = ind[::-1]

        ax1 = P.subplot(211)
        self.single_panel(ind[0], ind[1], def1[ind[0]], def2[ind[1]], scenarios=scenarios)

        if like_sign:
            ind = (0, 0)

        ax2 = P.subplot(212)
        self.single_panel(ind[1], ind[0], def1[ind[1]], def2[ind[0]], scenarios=scenarios)

        # Set common labels
        tr = self.margs[scenarios[0]].tr
        fig.text(0.52, 0.04, tr.to_tex(def1[0].name), ha='center', va='center')
        fig.text(0.04, 0.52, tr.to_tex(def2[0].name), ha='center', va='center', rotation='vertical')

        fig.subplots_adjust(left=0.2, right=0.92, bottom=0.1, top=0.98)

    def all_nuis_vs_wide(self):
        """
        compare all_nuis with all_wide
        """

        self.all_nuis_stack(self.comparison_defs['Re{c7}'], self.comparison_defs['Re{c9}'])
        P.savefig(self.out("all_nuis_wide_0_1"))

        self.all_nuis_stack(self.comparison_defs['Re{c7}'], self.comparison_defs['Re{c10}'], like_sign=True)
        P.savefig(self.out("all_nuis_wide_0_2"))

        self.all_nuis_stack(self.comparison_defs['Re{c9}'], self.comparison_defs['Re{c10}'], flip_order=True)
        P.savefig(self.out("all_nuis_wide_1_2"))

    def compare_scenarios(self, combinations):
        """
        compare multiple scenarios in overlays with filled and contoured regions
        """


#         self.comparison_bandwidths = {}

        for c in combinations:
            self.all_nuis_stack(self.comparison_defs['Re{c7}'], self.comparison_defs['Re{c9}'], scenarios=c)
            P.savefig(self.out("%s_0_1" % '_'.join(c)))

            self.all_nuis_stack(self.comparison_defs['Re{c7}'], self.comparison_defs['Re{c10}'], like_sign=True, scenarios=c)
            P.savefig(self.out("%s_0_2" % '_'.join(c)))

            self.all_nuis_stack(self.comparison_defs['Re{c9}'], self.comparison_defs['Re{c10}'], flip_order=True, scenarios=c)
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
        self.all_nuis_stack(self.comparison_defs['Re{c7}'], self.comparison_defs['Re{c9}'], scenarios=scenarios)
        P.savefig(self.out("%s_%s_0_1" % (scenarios)))

        self.all_nuis_stack(self.comparison_defs['Re{c7}'], self.comparison_defs['Re{c10}'], like_sign=True, scenarios=scenarios)
        P.savefig(self.out("%s_%s_0_2" % (scenarios)))

        self.all_nuis_stack(self.comparison_defs['Re{c9}'], self.comparison_defs['Re{c10}'], flip_order=True, scenarios=scenarios)
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

    def subleading_together(self):
        """
        Plot prior and posterior 1D for three subleading parameters with common prior
        """
        scenario = 'scI_posthep13'
        marg = self.margs[scenario]

        marg.use_nuisance = True
        marg.use_histogram = False
        marg.use_contours = False

        temp_font_size = matplotlib.rcParams['font.size']
        matplotlib.rcParams['font.size'] = 16

        class Prop(object):
            """properties of each figure"""
            def __init__(self, name, i, bandwidth=None, legend_label="",
                         style=dict(color=self.scen[scenario].c, linestyle='solid')):
                self.name = name
                # index of parameter in hdf5
                self.i = i
                # kde bandwidth
                self.bandwidth = bandwidth
                self.legend_label = legend_label
                self.style = style

        props = (Prop(12, 0.005), Prop(13, 0.015))

        prior_style=dict(color='black', linestyle='dashed', dashes=[4, 3])


        props = (Prop("B->K^*ll::A_perp^L_uncertainty@LargeRecoil",
                      12, bandwidth=0.005, legend_label=r"$\chi=\perp$",
                      style=dict(color='blue', marker='', linestyle='dashed')),
                 Prop("B->K^*ll::A_par^L_uncertainty@LargeRecoil", 23,  bandwidth=0.01,
                      legend_label=r"$\chi=\,\parallel$",
                      style=dict(color=self.scen[scenario].c, marker='', linestyle='-.')),
                 Prop("B->K^*ll::A_0^L_uncertainty@LargeRecoil", 21, bandwidth=0.015,
                      legend_label=r"$\chi=0$",
                      style=dict(color='green', linestyle='solid')))

        line_width = 1.5
        for p in props:
            p.style['linewidth'] = line_width

        prior_style = dict(color='black', linestyle='dotted', linewidth=line_width)
        prior_label = r"$\mathrm{prior}$"

        wide_figure()

        for i,p in enumerate(props):
            if p.bandwidth is not None:
                marg.use_histogram = False
                marg.kde_bandwidth = p.bandwidth
            else:
                marg.use_histogram = True

            # don't skip this plot
            marg.plot_prior = (i == 0)

            marg.one_dimensional(p.i, marginal_style=p.style, prior_style=prior_style,
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
        P.savefig(self.out('scI_subleading'))

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

        P.figure(figsize=(8,8))
        for k in self.scen.keys():
            for i, p1 in enumerate(self.pars):
                for j, p2 in enumerate(self.pars[self.pars.index(p1) + 1:]):
                    # draw SM point
                    P.plot(self.sm_point[self.defs[p1].i], self.sm_point[self.defs[p2].i], **self.sm_point_style)

                    # draw best fit points
                    for n, point in enumerate(self.scen[k].local_mode):
                        P.plot(point[self.defs[p1].i], point[self.defs[p2].i], \
                               **self.best_fit_points_style[n])

                    # add axis labels
                    P.xlabel(self.margs[k].tr.to_tex(p1))
                    P.ylabel(self.margs[k].tr.to_tex(p2))

                    # todo move to separate function
                    # plot/store density arrays
                    self.margs[k].use_histogram = not bool(self.scen[k].get_bandwidth(p1, p2))
                    self.margs[k].kde_bandwidth = self.scen[k].get_bandwidth(p1, p2)
                    self.scen[k].prob[(p1,p2)] = self.margs[k].two_dimensional(self.defs[p1].i, self.defs[p2].i)

                    # draw contours
                    P.clf()
                    CS = self.margs[k].contours_two(self.defs[p1].range, self.defs[p2].range, self.scen[k].prob[(p1,p2)], color=self.scen[k].c)
                    P.setp(CS.collections[1], alpha=self.scen[k].alpha)


                    P.savefig(self.out('%s_%d_%d' % (k, i, self.pars.index(p1) + j + 1)))

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

    def primed(self):
        """Scenario III marginals for C_i vs C'_i"""

        # overwrite ranges as they changed!
        self.defs['Re{c7}'] = ParameterDefinition(index=0, name='Re{c7}', min=-0.5, max=0.7)
        self.defs['Re{c9}'] = ParameterDefinition(index=1, name='Re{c9}', min=-6.5, max=5.5)
        self.defs['Re{c10}'] = ParameterDefinition(index=2, name='Re{c10}', min=-6, max=6)

        s = Scenario(os.path.join(self.input_base, 'pmc_scIII_posthep13.hdf5'), 'OrangeRed', bandwidth_default=0.005,
                                  sigma='1+2 sigma', two_sigma_color='LightSalmon', alpha=1, queue_output=False, crop_outliers=50,
                                  local_mode=[[ -0.337899, 3.30393, -4.48358, -0.0861297, 0.0512656, -0.837369],
                                              [0.502549, -4.1695, 3.52083, 0.0500648, -2.70563, -0.822219],
                                              [0.0115065, 3.28042, -1.57617, -0.424717, 3.38051, 2.46494],
                                              [0.119822, -3.94822, 1.98214, 0.431466, -3.46372, -2.24401]])
        m = plotScript.factory(self.command_template(s))


        # predictions in the SM
        primed_defs = (ParameterDefinition(index=3, name="Re{c7'}" , min=-0.6, max=0.6),
                       ParameterDefinition(index=4, name="Re{c9'}" , min=-6, max=6),
                       ParameterDefinition(index=5, name="Re{c10'}", min=-6, max=6),)
        for d in primed_defs:
            m.cuts[d.i] = d.range
        primed_predictions = (0, 0, 0)
        bandwidths = (0.01, 0.018, 0.013)
        mode_labels = (('A', ((-0.275, -0.1), (2.05, -1.25), (-3.5, -1.2))),
                       ('B', ((0.34, 0.02),  (-3.5, -2.6),   (4.2, -1.5))),
                       ('C', ((0.01, -0.37), (2.05, 2.5),   (-0.85, 2.25))),
                       ('D', ((0.1, 0.3), (-4.9, -5),  (2.3, -3.5))))

        fig = P.figure(figsize=(6, 6))

        for i in range(3):
            m.use_histogram = False
            m.kde_bandwidth = bandwidths[i]
            density = m.two_dimensional(i, i + len(self.pars))

            # draw contours
            P.clf()

            P.plot(self.sm_point[i], primed_predictions[i], **self.sm_point_style)
            xrange = (self.defs[primed_defs[i].name.replace("'",'')].min, self.defs[primed_defs[i].name.replace("'",'')].max)
            CS = m.contours_two(xrange, primed_defs[i].range, density, color=s.c)
            P.setp(CS.collections[1], alpha=s.alpha)

            P.setp(CS.collections[1], color=s.two_sigma_color)

            # indicate SM prediction
            P.plot(self.sm_point[i], primed_predictions[i], **self.sm_point_style)
            for p, (lab, loc) in zip(s.local_mode, mode_labels):
                P.plot(p[i], p[i+3], **self.best_fit_points_style[0])
                P.text(loc[i][0], loc[i][1], '$'+lab+'^{\prime}$')

            # don't whiten beyond 2 sigma, so ignore lowest contour fill
            #P.setp(CS.collections[0], alpha=0.0)
            CS.collections[0].remove()
            P.xlabel(m.tr.to_tex(self.pars[i]))
            P.ylabel(m.tr.to_tex(primed_defs[i].name))


            # 1:1 aspect ratio, but only if called before adjusting edges
            P.gca().set_aspect('equal')
            fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15)
            P.savefig(self.out('scIII_%d' % i))

    def nine_nine_prime(self):
        """Compare C9 vs C9' for posthep13 and posthep13 with hpqcd FF constraints"""

        sc_name = 'scII_posthep13'
        s = Scenario(os.path.join(self.input_base, 'pmc_%s.hdf5' % sc_name), 'OrangeRed', bandwidth_default=0.005,
                                  sigma='1+2 sigma', two_sigma_color='LightSalmon', alpha=1, queue_output=False, crop_outliers=50,
                                  local_mode=[[ 3.5, 1.1]])
        self.scen[sc_name] = s
        marg = plotScript.factory(self.command_template(s))
        self.margs[sc_name] = marg
        marg.use_histogram = True
        marg.kde_bandwidth = 0.001

        def9      = ParameterDefinition(name='Re{c9}',  min=-7.5, max=+7.5, index=0)
        def9prime = ParameterDefinition(name="Re{c9'}", min=-7.5, max=+7.5, index=1)

        P.figure()
        self.single_panel(0, 1, def9, def9prime, SM_point=False, local_mode=False, scenarios=(sc_name,))
        P.savefig(self.out(sc_name))
        return

        P.figure()
        density = marg.two_dimensional(def9.i, def9prime.i)
        self.__density_cache[(0,1, sc_name)] = density
        # draw contours
        P.clf()
        i = 1
        P.plot(self.sm_point[i], 0.0, **self.sm_point_style)
        xrange = (def9.min, def9.max)
        yrange = (def9prime.min, def9prime.max)
        CS = marg.contours_two(xrange, yrange, density, color=s.c)
        P.show()
        P.setp(CS.collections[1], alpha=s.alpha)

        P.setp(CS.collections[1], color=s.two_sigma_color)

        # indicate SM prediction
        P.plot(self.sm_point[i], 0.0, **self.sm_point_style)

        # don't whiten beyond 2 sigma, so ignore lowest contour fill
        #P.setp(CS.collections[0], alpha=0.0)
        CS.collections[0].remove()
        P.xlabel(marg.tr.to_tex(def9.name))
        P.ylabel(marg.tr.to_tex(def9prime.name))


        # 1:1 aspect ratio, but only if called before adjusting edges
        P.gca().set_aspect('equal')
#         fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15)
        P.savefig(self.out('harr'))
        return


#         shpqcd = Scenario(os.path.join(self.input_base, 'pmc_scII_posthep13hpqcd.hdf5'), 'OrangeRed', bandwidth_default=0.005,
#                                   sigma='1+2 sigma', two_sigma_color='LightSalmon', alpha=1, queue_output=False, crop_outliers=50,
#                                   local_mode=[[ -0.337899, 3.30393]])

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

class UncertaintyMarginals(object):
    """
    Plot 1D marginal distribution of SM uncertainty propagation and NP_unobs
    """
    def __init__(self, input_base, output_base, scenarios=['SM', 'NP'], n_samples=None):
        self.output_dir = os.path.join(output_base, "uncert")
        self.input_base = input_base
        self.ext = '.pdf'
        self.scenarios = scenarios

        self.read_data(n_samples)

#        ObservableProperties = namedtuple('ObservableProperties', 'name index range n_bins')


        self.observables = {'SM':tuple(),'NP':tuple()}

        SM_styles = dict(histo_style=dict(color='blue', linestyle='solid'),
                         credibility_style=dict(color='blue', alpha=0.4),)

        # arxiv_v1:
        # store name and index: name just for storing the file
        low_recoil_observables = (ObservableProperties(name='H_T_1_14_16', index=10, range=(0.9985, 1),
                                                       plot=True, n_ticks=4, mode_precision=4, **SM_styles),
                                  ObservableProperties(name='H_T_2_14_16', index=11, range=(-0.99, -0.97),
                                                       plot=True, n_ticks=5, **SM_styles),
                                  ObservableProperties(name='H_T_3_14_16', index=12, range=(-0.99, -0.97),
                                                       plot=True, n_ticks=5, **SM_styles),
                                  ObservableProperties(name='H_T_1_14+', index=36, mode_precision=4),
                                  ObservableProperties(name='H_T_2_14+', index=37, mode_precision=3),
                                  ObservableProperties(name='H_T_3_14+', index=38, mode_precision=3),
                                  ObservableProperties(name='H_T_1_16+', index=62, mode_precision=4),
                                  ObservableProperties(name='H_T_2_16+', index=63, mode_precision=3),
                                  ObservableProperties(name='H_T_3_16+', index=64, mode_precision=3),
                                  ObservableProperties(name='A_T_5_14_16', index=7, mode_precision=3),
                                  ObservableProperties(name='A_T_re_14_16', index=8, mode_precision=3),
                                  ObservableProperties(name='A_T_5_14+', index=33, mode_precision=3),
                                  ObservableProperties(name='A_T_re_14+', index=34, mode_precision=3),
                                  ObservableProperties(name='A_T_5_16+', index=59, mode_precision=3),
                                  ObservableProperties(name='A_T_re_16+', index=60, mode_precision=3),
                                  ObservableProperties(name='A_FB_14_16', index=2, mode_precision=3),
                                  ObservableProperties(name='F_L_14_16', index=3, mode_precision=3),
                                  ObservableProperties(name='A_FB_14+', index=29, mode_precision=3),
                                  ObservableProperties(name='F_L_14+', index=30, mode_precision=3),
                                  ObservableProperties(name='A_FB_16+', index=55, mode_precision=3),
                                  ObservableProperties(name='F_L_16+', index=56, mode_precision=3),
                                  )

        large_recoil_observables = (ObservableProperties(name='Br_1_6', index=27, range=(0, 1.e-6), n_bins=250, plot=True,
                                                         **SM_styles),
                                    ObservableProperties(name='F_L_1_6', index=29, range=(0, 1), plot=True, **SM_styles),
                                    ObservableProperties(name='A_T_re_1_6', index=34, range=(-0.2, 0.8), plot=True, **SM_styles),
                                    ObservableProperties(name='A_FB_2_4', index=2, mode_precision=3),
                                    ObservableProperties(name='A_FB_1_6', index=29, mode_precision=3),
                                   )

        self.observables['SM'] = (low_recoil_observables, large_recoil_observables)

        low_n_bins = 120
        NP_styles = dict(histo_style=dict(color='green', linestyle='dashed'),
                         credibility_style=dict(color='green', alpha=0.4))

        # arxiv_v1
        """
        o1 =                     (ObservableProperties(name='H_T_1_14_16', index=18, range=(0.9988, 1), plot=True,
                                                        n_ticks=4, n_bins=low_n_bins, mode_precision=4, **NP_styles),
                                  ObservableProperties(name='H_T_2_14_16', index=19, range=(-0.99, -0.97), plot=True,
                                                        n_ticks=5, n_bins=low_n_bins, **NP_styles),
                                  ObservableProperties(name='H_T_3_14_16', index=20, range=(-0.99, -0.97), plot=True,
                                                        n_ticks=5, n_bins=low_n_bins, **NP_styles),
                                  ObservableProperties(name='H_T_1_16+', index=21, plot=False, mode_precision=4),
                                  ObservableProperties(name='H_T_1_14+', index=24, plot=False, mode_precision=4),
                                  ObservableProperties(name='A_T_re_1_6', index=0, n_bins=low_n_bins,
                                                       range=(-0.1, 0.8), plot=True, **NP_styles))
        o2 =                     (ObservableProperties(name='Br_1_6', index=0, range=(0, 1e-6), plot=True,
                                                       n_ticks=6, n_bins=low_n_bins, **NP_styles),
                                  ObservableProperties(name='F_L_1_6', index=1, plot=True, n_bins=low_n_bins, **NP_styles),
                                  )
        o3 =                     (ObservableProperties(name='H_T_1_1_6', index=5, plot=False),
                                  ObservableProperties(name='H_T_2_1_6', index=11, plot=False))
        self.observables = {'SM':(low_recoil_observables, large_recoil_observables),
                            'NP':(o1, o2, o3)}
        """
        # need double tuple for compatibility with SM split across two files
        self.observables['NP'] = ((ObservableProperties(name='H_T_1_14_16', index=36, range=(0.9988, 1), plot=True,
                                                        n_ticks=4, n_bins=low_n_bins, mode_precision=4, **NP_styles),
                                  ObservableProperties(name='H_T_2_14_16', index=37, range=(-0.99, -0.97), plot=True,
                                                        n_ticks=5, n_bins=low_n_bins, **NP_styles),
                                  ObservableProperties(name='H_T_3_14_16', index=38, range=(-0.99, -0.97), plot=True,
                                                        n_ticks=5, n_bins=low_n_bins, **NP_styles),
                                  ObservableProperties(name='H_T_1_16+', index=39, plot=False, mode_precision=4),
                                  ObservableProperties(name='H_T_1_14+', index=42, plot=False, mode_precision=4),
                                    ObservableProperties(name='A_T_re_1_6', index=30, n_bins=50,
                                                       range=(-0.1, 0.8), plot=True, **NP_styles),
                                   # use K*ll, not Kll!
                                    ObservableProperties(name='Br_1_6', index=45, range=(0, 1e-6), plot=True,
                                                       n_ticks=6, n_bins=low_n_bins, **NP_styles),
                                    ObservableProperties(name='F_L_1_6', index=46, plot=True, n_bins=low_n_bins, **NP_styles),
                                    ObservableProperties(name='H_T_1_1_6', index=34, plot=False),
                                    ObservableProperties(name='H_T_2_1_6', index=35, plot=False)),)

        self.table_single_args = {'SM':dict(skip_SM_prediction=False), 'NP':dict(skip_SM_prediction=True)}

        self.__define_scale_factors()

        self.__observable_props()

    def __define_scale_factors(self):

        if 'SM' not in self.scenarios:
            return

        # log10
        self.uncert['SM'][0].scale_factors = {
        "B->Kll::BR@LowRecoil":7,
        "B->K^*ll::BR@LowRecoil":7,
        "B->K^*ll::J_1s@LowRecoil":8,
        "B->K^*ll::J_1c@LowRecoil":8,
        "B->K^*ll::J_2s@LowRecoil":8,
        "B->K^*ll::J_2c@LowRecoil":8,
        "B->K^*ll::J_3@LowRecoil":8,
        "B->K^*ll::J_4@LowRecoil":8,
        "B->K^*ll::J_5@LowRecoil":8,
        "B->K^*ll::J_6s@LowRecoil":7,
        "B->K^*ll::J_7@LowRecoil":12,
        "B->K^*ll::J_8@LowRecoil":11,
        "B->K^*ll::J_9@LowRecoil":11,
        "B->K^*ll::A_T^im@LowRecoil":4,
        "B->K^*ll::H_T^4@LowRecoil":3,
        "B->K^*ll::H_T^5@LowRecoil":3,
        }

        self.uncert['SM'][1].scale_factors = {
        "B->Kll::BR@LargeRecoil":7,
        "B->K^*ll::BR@LargeRecoil":7,
        "B->K^*ll::J_1s@LargeRecoil":8,
        "B->K^*ll::J_1c@LargeRecoil":7,
        "B->K^*ll::J_2s@LargeRecoil":8,
        "B->K^*ll::J_2c@LargeRecoil":7,
        "B->K^*ll::J_3@LargeRecoil":10,
        "B->K^*ll::J_4@LargeRecoil":8,
        "B->K^*ll::J_5@LargeRecoil":8,
        "B->K^*ll::J_6s@LargeRecoil":8,
        "B->K^*ll::J_7@LargeRecoil":9,
        "B->K^*ll::J_8@LargeRecoil":9,
        "B->K^*ll::J_9@LargeRecoil":11,
#        "B->K^*ll::A_T^2@LargeRecoil":2,
        "B->K^*ll::A_T^im@LargeRecoil":3,
#        "B->K^*ll::H_T^4@LargeRecoil":2,
        "B->K^*ll::H_T^5@LargeRecoil":3,
        }

    def __observable_props(self):
        """
        Set ranges and precision for individual observables.
        """
        for scen in self.uncert.iterkeys():
            for regime in self.observables[scen]:
                u = self.uncert[scen][self.observables[scen].index(regime)]
                for o in regime:
                    if o.range:
                        u.observable_ranges[o.index] = o.range
                    if o.mode_precision:
                        u.mode_precision[o.index] = o.mode_precision

    def out(self, name, ext=None):
        """ Create output file name"""
        ext = self.ext if ext is None else ext
        return os.path.join(self.output_dir, "uncert_prop_%s%s" % (name, ext))

    def plot(self):
        """
        Plot 1D probability distributions, with overlay of SM and NP if available
        """


        # determine which observable appears in both scenarios
        observable_names = {}
        all_observable_names = Set()
        for scen in self.uncert.iterkeys():
            scen_observable_names = Set()
            for regime in self.observables[scen]:
                for o in regime:
                    if not o.plot:
                        continue
                    scen_observable_names.add(o.name)
                    all_observable_names.add(o.name)
            observable_names[scen] = scen_observable_names


        multiples = observable_names[self.uncert.keys()[0]]
        for scen in self.uncert.keys()[1:]:
            multiples = multiples & observable_names[scen]
        singles = all_observable_names - multiples

        print("Multiples: %s" % multiples)
        print("Singles: %s" % singles)

        multiple_uncerts = {}
        for name in multiples:
            multiple_uncerts[name] = []

        # now plot singles
        for scen in self.uncert.iterkeys():
            for regime in self.observables[scen]:
                i = self.observables[scen].index(regime)
                u = self.uncert[scen][i]
                for o in regime:
                    if o.name in singles:
                        wide_figure(top=0.93)
                        u.n_bins = o.n_bins
                        u.single_plot(obs_index=o.index, n_ticks=o.n_ticks, y_labels=o.y_labels)
                        P.savefig(self.out("%s_%s" % (scen, o.name)))
                    elif o.name in multiples:
                        multiple_uncerts[o.name].append((u, o))
                    else:
                        pass

        # now plot multiples
        for name in multiples:
            wide_figure(ratio=16/9.0, left=0.06, right=0.93, bottom=0.2)
            for u,o in multiple_uncerts[name]:
                u.n_bins = o.n_bins
                args = dict()
                if o.histo_style:
                    args['histo_style'] = o.histo_style
                if o.credibility_style:
                    args['credibility_style'] = o.credibility_style
                if o.n_ticks:
                    args['n_ticks'] = o.n_ticks
                args['y_labels'] = o.y_labels
                ret_val = u.single_plot(obs_index=o.index, **args)
            P.savefig(self.out("overlay_%s" % o.name))

    def plot_binned_predictions(self, modes, one_sigma_intervals, two_sigma_intervals, (q2_min, q2_max, step)=(1,6,1),
                                line_style=dict(color='black'), one_sigma_style=OrderedDict(color='green', alpha=1),
                                two_sigma_style=OrderedDict(color='yellow', alpha=1),
                                mode_thickness=0.001):
        """
        Plot integrated observables as function of q^2

        args:
        mode -- array of size N
        one_sigma -- array of size N, each a (min, max) interval containing 68%
        two_sigma -- array of size N, each a (min, max) interval containing 95%
        q2_min -- minimum value on x-axis
        q2_max -- maximum value on x-axis
        step -- difference between two left bin edges: fixed binning
        """

        # validate input
        n = int((q2_max - q2_min) / step)
        assert(len(modes) == n)
        assert(len(one_sigma_intervals) == n)
        assert(len(two_sigma_intervals) == n)

        """
        s1 = one_sigma_style.copy()
        s2 = two_sigma_style.copy()
#        s1['edgecolor'] = s2['edgecolor'] ='none'
#        s1['linewidth'] = s2['linewidth'] = 0.0
        s1['fill'] = s2['fill'] = True
        s1['snap'] = s2['snap'] = True
        bb = matplotlib.patches.BoxStyle('round', pad=0.0, rounding_size=0.0)
        s1['boxstyle'] = s2['boxstyle'] = bb

        rect = matplotlib.patches.FancyBboxPatch
#        rect = matplotlib.patches.Rectangle
        """

        ax = P.gca()
        for i, q2 in enumerate(np.arange(q2_min, q2_max, step)):

            # line is too long, hand adjust its length
#            P.plot(((2-0.997)*q2, q2 + 0.9945*step), [modes[i]]*2, **line_style)

            """
            # unsuccessfull attempt to turn off rounding the corners
            ax.add_patch(rect((q2, modes[i]), step, one_sigma_intervals[i][1] - modes[i], **s1))
            ax.add_patch(rect((q2, one_sigma_intervals[i][1]), step, two_sigma_intervals[i][1] -  one_sigma_intervals[i][1], **s2))

#            ax.add_patch(p)
#            bb = p.get_bbox()
#            bb.set_pad(0)
#            bb.set_boxstyle("square")
            """
            # no edge color to avoid rounded edge corners
            P.fill_between((q2, q2 + step), [one_sigma_intervals[i][0]]*2, [one_sigma_intervals[i][1]]*2, edgecolor='none', clip_on=False, **one_sigma_style)
            P.fill_between((q2, q2 + step), [one_sigma_intervals[i][0]]*2, [two_sigma_intervals[i][0]]*2, edgecolor='none', clip_on=False, **two_sigma_style)
            P.fill_between((q2, q2 + step), [one_sigma_intervals[i][1]]*2, [two_sigma_intervals[i][1]]*2, edgecolor='none', clip_on=False, **two_sigma_style)

            P.fill_between((q2, q2 + step), [modes[i] - mode_thickness]*2, [modes[i] + mode_thickness]*2, edgecolor='none', clip_on=False, **line_style)

        P.xlim(q2_min, q2_max)

    def plot_large_recoil_single_bins(self):
        modes = np.arange(1, 6, 1)
        one_sigma_intervals = [(x - 0.2, x + 0.2) for x in modes]
        two_sigma_intervals = [(x - 0.4, x + 0.4) for x in modes]
        self.plot_binned_predictions(modes=modes, one_sigma_intervals=one_sigma_intervals, two_sigma_intervals=two_sigma_intervals)
        P.ylim(0, 7)
        P.savefig(self.out("harr"))

        EvolutionProperties = namedtuple('EvolutionProperties', 'name scen file_index indices y_range')

        # file indices of arxiv_v1
        """
        props = (EvolutionProperties(name='A_T_5', scen='NP',  file_index=1, indices=range(3,8), y_range=(0, 0.5)),
                 EvolutionProperties(name='A_T_re', scen='NP', file_index=0, indices=(3,6,9,12,15), y_range=(-1, 1)),
                 EvolutionProperties(name='A_T_3', scen='NP',  file_index=0, indices=(4,7,10,13,16), y_range=(0, 1.5)),
                 EvolutionProperties(name='H_T_1', scen='NP',  file_index=2, indices=range(0,5), y_range=(-0.5, 1)),
                 EvolutionProperties(name='H_T_2', scen='NP',  file_index=2, indices=range(6,11), y_range=(-1, 0.5)),
                 EvolutionProperties(name='A_T_4', scen='NP',  file_index=0, indices=(5,8,11,14,17), y_range=(0, 3)))
        """
        props = (EvolutionProperties(name='A_T_5', scen='NP',  file_index=0, indices=range(3,5*6,6), y_range=(0, 0.5)),
                 EvolutionProperties(name='A_T_re', scen='NP', file_index=0, indices=range(0,5*6,6), y_range=(-1, 1)),
                 EvolutionProperties(name='A_T_3', scen='NP',  file_index=0, indices=range(1,5*6,6), y_range=(0, 1.5)),
                 EvolutionProperties(name='H_T_1', scen='NP',  file_index=0, indices=range(4,5*6,6), y_range=(-0.5, 1)),
                 EvolutionProperties(name='H_T_2', scen='NP',  file_index=0, indices=range(5,5*6,6), y_range=(-1, 0.5)),
                 EvolutionProperties(name='A_T_4', scen='NP',  file_index=0, indices=range(2,5*6,6), y_range=(0, 3)))

        for p in props:
            wide_figure(x_size=6, ratio=1, left=0.2, bottom=0.13)
            modes = []
            one_sigma_intervals = []
            two_sigma_intervals = []
            for i in p.indices:
                u = self.uncert[p.scen][p.file_index]
                template, mode, intervals = u.single_plot(i)
                P.clf()
                P.ylabel(plotScript.Translator.to_tex(u.observable_names[i]))
                modes.append(mode)
                one_sigma_intervals.append(intervals[0])
                two_sigma_intervals.append(intervals[1])

            self.plot_binned_predictions(modes=modes, one_sigma_intervals=one_sigma_intervals, two_sigma_intervals=two_sigma_intervals,
                                         mode_thickness=(p.y_range[1] - p.y_range[0]) / 500.0)
            # turn on minor ticks on y-axis
            ax = P.gca()
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            P.xlabel(r"$q^2 [\mathrm{GeV}^2]$")

            P.ylim(p.y_range)
            P.savefig(self.out(p.name))

    def read_data(self, n_samples):

        n_bins = 180

        inputs = {}
        self.uncert = {}

        if 'SM' in self.scenarios:
            inputs['SM'] = ('sc1_SM_lowRec_unc_merge.hdf5', 'sc1_SM_largeRec_unc_merge.hdf5',)
#            inputs['SM'] = ('sc1_SM_largeRec_unc_merge.hdf5',) # only for last minute changes
            self.uncert['SM'] = []

        if 'NP' in self.scenarios:
            inputs['NP'] = ('sc1_NP_unobs_unc.hdf5',)
            # data was in three separate files for arxiv_v1
#            inputs['NP'] = ('sc1_NP_unobs_unc.hdf5', 'sc1_NP_unobs_unc_extra.hdf5', 'sc1_NP_unobs_unc_H_T_largeRec.hdf5')
            self.uncert['NP'] = []

        for scen, input_files in inputs.iteritems():
            for f in input_files:
                # need list of strings
                cmd_line = self.input_base + f
                cmd_line += " --1D-bins %d" % n_bins
                if n_samples:
                    cmd_line += " --select 0 %d" % n_samples
                u = plotUncertainty.factory(cmd_line.split())
                u.print_uncertainty = False

                self.uncert[scen].append(u)

    def table(self):
        """Create latex table of uncertainties"""

        for scen, uncerts in self.uncert.iteritems():
            print(uncerts[0].observable_names)
            f = open(os.path.join(self.output_dir, 'uncertainty_%s_table.template.tex' % scen))
            input_template = f.read()
            f.close()

            for regime in uncerts:
                input_template = regime.plot_all(input_template=input_template, single_args=self.table_single_args[scen])

            f = open(os.path.join(self.output_dir, 'uncertainty_%s_table.tex' % scen), 'w')
            f.write(input_template)
            f.close()

        return

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

def fall2013():

    # set up object
    input_base, output_base = input_output()
    marg = MarginalContours(input_base, output_base, max_samples=None)

    marg.nine_nine_prime()
    return

    marg.scen['scI_posthep13'] = Scenario(os.path.join(input_base, 'pmc_scI_posthep13.hdf5'), 'OrangeRed', bandwidth_default=0.005,
                                    sigma='1+2 sigma', two_sigma_color='LightSalmon', alpha=1, queue_output=False, crop_outliers=50,
                                    local_mode=[[-0.342938, 3.94893, -4.61573],
                                                [0.505892, -5.00182, 4.50871]])
    marg.read_data()
    marg.subleading_together()
    return
    marg.scen['scI_quim1'] = Scenario(os.path.join(input_base, 'pmc_scI_quim1.hdf5'), 'Blue', bandwidth_default=0.005,
                                      sigma='1+2 sigma', two_sigma_color='LightBlue', alpha=1, queue_output=False, crop_outliers=50,
                                      local_mode=[[-0.345514, 2.99263, -4.16734],
                                                  [ 0.509072, -4.02532, 4.22568]])
    marg.scen['scI_all_nuis'] = Scenario(os.path.join(input_base, 'pmc_scI_all_nuis.hdf5'), 'Grey', bandwidth_default=0.005,
                                         sigma='1+2 sigma', two_sigma_color='LightGrey', alpha=1, crop_outliers=200,
                                      local_mode=[[-0.294991910838,    3.731820480717,  -4.140554057902],
                                                  [ 0.41787049285,  -4.639111764728,  3.994616452063]])
    marg.read_data()

    # define bandwidths for each individual plot
    bw = 0.0045
    marg.comparison_bandwidths[('scI_posthep13', 1, 0, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # top
    marg.comparison_bandwidths[('scI_posthep13', 0, 1, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # bottom
    marg.comparison_bandwidths[('scI_posthep13', 0, 0, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # bottom
    marg.comparison_bandwidths[('scI_posthep13', 1, 1, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # top
    marg.comparison_bandwidths[('scI_posthep13', 1, 0, 'Re{c9}', 'Re{c10}')] = bw * 1.7  # bottom
    marg.comparison_bandwidths[('scI_posthep13', 0, 1, 'Re{c9}', 'Re{c10}')] = bw * 1.4  # top

    marg.comparison_bandwidths[('scI_quim1', 1, 0, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # top
    marg.comparison_bandwidths[('scI_quim1', 0, 1, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # bottom
    marg.comparison_bandwidths[('scI_quim1', 0, 0, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # bottom
    marg.comparison_bandwidths[('scI_quim1', 1, 1, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # top
    marg.comparison_bandwidths[('scI_quim1', 1, 0, 'Re{c9}', 'Re{c10}')] = bw * 1.5  # bottom
    marg.comparison_bandwidths[('scI_quim1', 0, 1, 'Re{c9}', 'Re{c10}')] = bw * 1.4  # top

    marg.comparison_bandwidths[('scI_all_nuis', 1, 0, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # top
    marg.comparison_bandwidths[('scI_all_nuis', 0, 1, 'Re{c7}', 'Re{c9}')] = bw * 1.4  # bottom
    marg.comparison_bandwidths[('scI_all_nuis', 0, 0, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # bottom
    marg.comparison_bandwidths[('scI_all_nuis', 1, 1, 'Re{c7}', 'Re{c10}')] = bw * 1.6  # top
    marg.comparison_bandwidths[('scI_all_nuis', 1, 0, 'Re{c9}', 'Re{c10}')] = bw * 1.5  # bottom
    marg.comparison_bandwidths[('scI_all_nuis', 0, 1, 'Re{c9}', 'Re{c10}')] = bw * 1.4  # top

    # ##
    # ACTIONS
    # ##
    marg.single_scenario()
    combinations = (('scI_all_nuis',), ('scI_quim1',), ('scI_posthep13',),
                    ('scI_all_nuis', 'scI_posthep13'), ('scI_all_nuis', 'scI_quim1'),
                    ('scI_posthep13', 'scI_quim1'))
    marg.compare_scenarios(combinations)
    marg.subleading_separate()

    # scIII plot
    marg.primed()

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

    fall2013()
    # ##
    # ACTIONS
    # ##


#                            ignore_scenarios=('all_wide',)),'Vll_lowRec', 'Vll_largeRec', 'Pll'))

#     marg.one_dim_nuisance_KMPW()
    # marg.subleading_together()
#     marg.single_scenario(); marg.overlays()
#     marg.scI_all_vs_excl()
#     marg.compare_scenarios()
#    marg.all_nuis_vs_wide()
#     marg.credibility_regions()

#    pull(output_base, mode=0, SM=False)

#    unc = UncertaintyMarginals(output_base, scenarios=['NP'], n_samples=None)
#    unc.plot()
#    unc.plot_large_recoil_single_bins()
#    unc.table()
