#! /usr/bin/python
"""
Plot several distributions with their contours for paper
"""

import plotScript
import plotUncertainty
import pylab as P
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.patches
from matplotlib.lines import Line2D
import numpy as np
import os
import itertools
from collections import OrderedDict, defaultdict, namedtuple
from sets import Set

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
    def __init__(self, file, color, nbins=150, alpha=0.4, sigma='2 sigma', bandwidth_default=None, two_sigma_color=None):
        self.f = file
        self.c = color
        self.prob = {}
        self.nbins = nbins
        self.alpha = alpha
        self.sigma = sigma
        self.__bandwidth = {}
        self.__bandwidth_default = bandwidth_default
        self.two_sigma_color = two_sigma_color

    def get_bandwidth(self, par1, par2):
        try:
            return self.__bandwidth[(par1,par2)]
        except KeyError:
            return self.__bandwidth_default

    def set_bandwidth(self, par1, par2, value):
        self.__bandwidth[(par1, par2)] = value

class MarginalContours(object):

    def __init__(self, input_base, output_base, ext='.pdf', max_samples=None, ignore_scenarios=None):

        self.output_base = os.path.join(output_base, "contour")
        self.ext = ext

        self.pars = ['Re{c7}', 'Re{c9}', 'Re{c10}']

        self.defs = OrderedDict()
        #(index, (min, max))
        self.defs['Re{c7}'] = plotScript.ParameterDefinition(index=0, name='Re{c7}', min=-1, max=+1)
        self.defs['Re{c9}'] = plotScript.ParameterDefinition(index=1, name='Re{c9}', min=-15, max=15)
        self.defs['Re{c10}'] = plotScript.ParameterDefinition(index=2, name='Re{c10}', min=-15, max=15)

        self.input_base = input_base

        self.scen = OrderedDict()

        self.scen['Pll'] = Scenario(self.input_base + 'sc1_BPll.hdf5', 'DarkGoldenRod', bandwidth_default=0.007)
        self.scen['Pll'].set_bandwidth('Re{c9}', 'Re{c10}', 0.01)

        # bw=0.01 is slighly enlarging the regions, but ensures smooth contours, and no freckles
        self.scen['Vll_largeRec'] = Scenario(self.input_base + 'sc1_BVll_largeRec.hdf5', 'blue', bandwidth_default=0.01)
#        self.scen['Vll_largeRec'].set_bandwidth('Re{c7}', 'Re{c9}', 0.003)
#        self.scen['Vll_largeRec'].set_bandwidth('Re{c7}', 'Re{c10}', 0.003)
#        self.scen['Vll_largeRec'].set_bandwidth('Re{c9}', 'Re{c10}', 0.005)

        self.scen['Vll_lowRec'] = Scenario(self.input_base + 'sc1_BVll_lowRec.hdf5', 'LimeGreen', alpha=0.7, bandwidth_default=0.01)

        # bandwidth tuned for good looking bimodal plot; zoom ins have different bandwidths!
        self.scen['all_wide'] = Scenario(self.input_base + 'sc1_all_wide.hdf5', 'Black', bandwidth_default=0.005, sigma='1+2 sigma')
#        self.scen['all_2sigma'] = Scenario(self.input_base + 'sc1_all_nuis.hdf5', 'LightSalmon', alpha=1)
        self.scen['all_nuis'] = Scenario(self.input_base + 'sc1_all_nuis.hdf5', 'OrangeRed', bandwidth_default=0.005,
                                    sigma='1+2 sigma', two_sigma_color='LightSalmon', alpha=1)

        # remove some for testing and increasing speed
        if ignore_scenarios:
            for s in ignore_scenarios:
                del(self.scen[s])

        print("Operating on: %s" % str(self.scen.keys()))

        self.sm_point = [-0.32741917, +4.27584794, -4.15077942]
        self.sm_point_style = dict(marker = 'D', markersize = 8, color = 'black')

        self.best_fit_points = [[-0.294991910838,    3.731820480717,  -4.140554057902],
                                [ 0.41787049285,  -4.639111764728,  3.994616452063]]
        best_fit_point_style = dict(self.sm_point_style)
        best_fit_point_style['marker'] = 'x'
        best_fit_point_style['markeredgewidth'] = 3.0

        self.best_fit_points_style = [best_fit_point_style]*2

        self.max_samples = max_samples
        self.read_data()

        # store bandwidths
        # format: (i, j, par1, par2)
        self.all_nuis_wide_bandwidths = {}

    def __get_bandwidth__(self, i, j, par1, par2):
        try:
            return self.all_nuis_wide_bandwidths[(i, j, par1, par2)]
        except KeyError:
            return None

    def out(self, name):
        """ Create output file name"""
        return os.path.join(self.output_base, name) + self.ext

    ###
    # template for single plots
    ###
    def command_template(self, input, nbins):
        cmd_template = ''
        cmd_template += ' %s' % input
    #    cmd_template += ' --single-2D %d %d' % (par1, par2)
        cmd_template += ' --pmc-crop-outliers 200'
        cmd_template += ' --2D-bins %d' % nbins
        if self.max_samples is not None:
            cmd_template += ' --select 0 %d' % self.max_samples
        cmd_template += ' --contours'

        for p in self.defs.itervalues():
            cmd_template += ' --cut %d %s %s' % (p.i, p.min, p.max)

        return cmd_template.split()

    def read_data(self):

        nbins = 75
        self.margs = {}
        for k in self.scen.keys():
            self.margs[k] = plotScript.factory(self.command_template(self.scen[k].f, self.scen[k].nbins))

    def single_panel(self, def1, def2, bandwidth=None, SM_point=True, local_mode=True):
        """ Single marginal plot to compare wide with all nuis"""

        # parameter indices in EOS output file
        i = self.pars.index(def1.name)
        j = self.pars.index(def2.name)

        scenarios = ('all_wide', 'all_nuis')

        prob_density = {}

        # apply range cuts and compute density
        for s in scenarios:
            self.margs[s].cuts[i] = def1.range
            self.margs[s].cuts[j] = def2.range
            if bandwidth is not None:
                self.margs[s].use_histogram = False
                self.margs[s].kde_bandwidth = bandwidth
                print("bandwidth = %g " %bandwidth)
            else:
                self.margs[s].use_histogram = True
            prob_density[s] = self.margs[s].two_dimensional(i, j)

        P.cla()

        for s in scenarios:
            self.margs[s].contours_two(def1.range, def2.range, prob_density[s],
                                       color=self.scen[s].c, line=True if s == 'all_wide' else False)

        if SM_point:
            P.plot(self.sm_point[i], self.sm_point[j], **self.sm_point_style)
        if local_mode:
            for style, p in zip(self.best_fit_points_style, self.best_fit_points):
                P.plot(p[i], p[j], **style)

        ax = P.gca()

        ax.xaxis.set_major_locator(ticker.LinearLocator(def1.n_major_ticks))
        ax.set_xlim(def1.range)

        ax.yaxis.set_major_locator(ticker.LinearLocator(def2.n_major_ticks))
        ax.set_ylim(def2.range)

    def all_nuis_stack(self, def1, def2, like_sign=False, flip_order=False):
        """
        Plot two panels, one on top of the other with
        marginal distribution for two parameters.

        Keyword arguments:
        def1, def2 -- ParameterDefinition of both parameters.
        like_sign -- If true, use +,+ and -,-. Default: false, then use -+, +-.
        """

        x_size = 5
        fig = P.figure(figsize=(x_size, 2 * x_size))

        # use -+ or +-?
        ind = (1, 0)
        if like_sign:
            ind = (1, 1)
        if flip_order:
            ind = ind[::-1]

        ax1 = P.subplot(211)
        self.single_panel(def1[ind[0]], def2[ind[1]], self.__get_bandwidth__(ind[0], ind[1], def1[0].name, def2[0].name))

        if like_sign:
            ind = (0, 0)

        ax2 = P.subplot(212)
        self.single_panel(def1[ind[1]], def2[ind[0]], self.__get_bandwidth__(ind[1], ind[0], def1[0].name, def2[0].name))

        # Set common labels
        fig.text(0.52, 0.04, plotScript.Translator.to_tex(def1[0].name), ha='center', va='center')
        fig.text(0.04, 0.52, plotScript.Translator.to_tex(def2[0].name), ha='center', va='center', rotation='vertical')

        fig.subplots_adjust(left=0.2)
#        fig.subplots_adjust(right=right_adjust)
        fig.subplots_adjust(top=0.96)
        fig.subplots_adjust(bottom=0.1)

    def all_nuis_vs_wide(self):
        """
        compare all_nuis with all_wide
        """

        tight_defs = {}
        tight_defs['Re{c7}'] = (plotScript.ParameterDefinition(name='Re{c7}', min=+0.0, max=+0.7, index=self.pars.index('Re{c7}'), n_major_ticks=3),
                                plotScript.ParameterDefinition(name='Re{c7}', min=-0.7, max=0.0, index=self.pars.index('Re{c7}'), n_major_ticks=3))
        tight_defs['Re{c9}'] = (plotScript.ParameterDefinition(name='Re{c9}', min=+1.5, max=+7.5, index=self.pars.index('Re{c9}'), n_major_ticks=3),
                                plotScript.ParameterDefinition(name='Re{c9}', min=-8, max=-2, index=self.pars.index('Re{c9}'), n_major_ticks=3))
        tight_defs['Re{c10}'] = (plotScript.ParameterDefinition(name='Re{c10}', min=+1, max=+6, index=self.pars.index('Re{c10}'), n_major_ticks=3),
                                 plotScript.ParameterDefinition(name='Re{c10}', min=-6, max=-1, index=self.pars.index('Re{c10}'), n_major_ticks=3))

        # define bandwidths for each individual plot
        bw = 0.02
        self.all_nuis_wide_bandwidths[(1, 0, 'Re{c7}', 'Re{c9}')]  = bw * 1.4   # top
        self.all_nuis_wide_bandwidths[(0, 1, 'Re{c7}', 'Re{c9}')]  = bw * 1.4   # bottom
        self.all_nuis_wide_bandwidths[(0, 0, 'Re{c7}', 'Re{c10}')] = bw * 1.6   # bottom
        self.all_nuis_wide_bandwidths[(1, 1, 'Re{c7}', 'Re{c10}')] = bw * 1.6   # top
        self.all_nuis_wide_bandwidths[(1, 0, 'Re{c9}', 'Re{c10}')] = bw * 1.5   # bottom
        self.all_nuis_wide_bandwidths[(0, 1, 'Re{c9}', 'Re{c10}')] = bw * 1.4   # top

        self.all_nuis_stack(tight_defs['Re{c7}'], tight_defs['Re{c9}'])
        P.savefig(self.out("all_nuis_wide_0_1"))

        self.all_nuis_stack(tight_defs['Re{c7}'], tight_defs['Re{c10}'], like_sign=True)
        P.savefig(self.out("all_nuis_wide_0_2"))

        self.all_nuis_stack(tight_defs['Re{c9}'], tight_defs['Re{c10}'], flip_order=True)
        P.savefig(self.out("all_nuis_wide_1_2"))

        return

        """

        for p1 in self.pars:
            for p2 in self.pars[i + 1:]:
                panel = 0

                prob_density = {}
                # apply range cuts and compute density
                for s in scenarios:
                    self.margs[s].cuts[i] = tight_defs[p1][panel].range
                    self.margs[s].cuts[j] = tight_defs[p2][panel].range
                    prob_density[s] = self.margs[s].two_dimensional(i, j)

                P.clf()

                for s in scenarios:
                    self.margs[s].contours_two(tight_defs[p1][panel].range, tight_defs[p2][panel].range, prob_density[s], color=self.scen[s].c)

                P.subplot(212)
                panel = 1



        self.margs['all_wide'].contours_two(tight_defs['c7-'].range, tight_defs['c9+'].range, self.scen['all_wide'].prob[(0, 1)], color=self.scen['all_wide'].c)
        self.margs['all_2sigma'].contours_two(tight_defs['c7-'].range, tight_defs['c9+'].range, self.scen['all_2sigma'].prob[(0, 1)], color=self.scen['all_2sigma'].c)
        P.subplot(212)
        self.margs['all_wide'].cuts[self.pars.index('Re{c7}')] = tight_defs['c7+'].range
        self.margs['all_wide'].cuts[self.pars.index('Re{c9}')] = tight_defs['c9-'].range

        self.margs['all_wide'].two_dimensional(0, 1)

        """

    def one_dim_nuisance_KMPW(self):
        """
        Plot prior and posterior 1D for two different subscenarios in one plot.
        """
        """
        plotScript.py sc1_BPll_parameter_samples_12.hdf5_merge  --nuisance --pmc-crop-outliers 200 --single-1D 12 --use-KDE --bandwidth 0.004
        plotScript.py sc1_BPll_parameter_samples_12.hdf5_merge  --nuisance --pmc-crop-outliers 200 --single-1D 13 --use-KDE --bandwidth 0.04
        """

        scenarios = ('all_nuis', 'Pll')

        parameters = ("B->K::F^p(0)@KMPW2010", "B->K::b^p_1@KMPW2010")

        bandwidths = {}
        bandwidths[('Pll', "B->K::F^p(0)@KMPW2010")] = 0.004
        bandwidths[('Pll', "B->K::b^p_1@KMPW2010")] = 0.04
        bandwidths[('all_nuis', "B->K::F^p(0)@KMPW2010")] = 0.003
        bandwidths[('all_nuis', "B->K::b^p_1@KMPW2010")] = 0.04

        par_indices = {}
        par_indices[('Pll', "B->K::F^p(0)@KMPW2010")] = 12
        par_indices[('Pll', "B->K::b^p_1@KMPW2010")] = 13
        par_indices[('all_nuis', "B->K::F^p(0)@KMPW2010")] = 13
        par_indices[('all_nuis', "B->K::b^p_1@KMPW2010")] = 14

        marginal_styles = {}
        marginal_styles['Pll'] = dict(color=self.scen['Pll'].c, linestyle='dashed')
        marginal_styles['all_nuis'] = dict(color=self.scen['all_nuis'].c, linestyle='solid')

        line_width = 1.5
        for m in marginal_styles.values():
            m['linewidth'] = line_width

        prior_style = dict(color='black', linestyle='dotted', linewidth=line_width)

        x_labels = {}
        x_labels["B->K::b^p_1@KMPW2010"] = '$b_1^+$'
        x_labels["B->K::F^p(0)@KMPW2010"] = '$f_{+}(0)$'

        x_size = 8; y_size = 6

        for p in parameters:

            fig = P.figure(figsize=(x_size, y_size))
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.08)
            fig.subplots_adjust(right=0.95)
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(bottom=0.15)

            for n_sc, s in enumerate(scenarios):
                i = par_indices[(s, p)]

                try:
                    bw = bandwidths[(s, p)]
                except KeyError:
                    bw = None
                if bw is not None:
                    self.margs[s].use_histogram = False
                    self.margs[s].kde_bandwidth = bw
                else:
                    self.margs[s].use_histogram = True

                # don't skip this plot
                self.margs[s].use_nuisance = True
                self.margs[s].use_contours = False
                self.margs[s].plot_prior = (n_sc == 0)
#                self.margs[s].use_histogram = True

                self.margs[s].one_dimensional(i, marginal_style=marginal_styles[s], prior_style=prior_style)

            P.xlabel(x_labels[p])
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
#            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            P.setp(ax.get_yticklabels(), visible=False)

            P.savefig(self.out('nuis_1D_FF_%s' % p))

    def one_dim_nuisance_BZ(self):
        """
        Plot prior and posterior 1D for all three BZ form factors and one scenario with common prior
        """
        scenario = 'all_nuis'

        class Prop(object):
            def __init__(self, par_name, par_index, bandwidth=None, legend_label="",
                         style=dict(color=self.scen[scenario].c, linestyle='solid')):
                self.par_name = par_name
                self.par_index = par_index
                self.bandwidth = bandwidth
                self.legend_label = legend_label
                self.style = style


        props = (Prop("B->K^*::a1_uncertainty@BZ2004", 10, bandwidth=0.01,
                      legend_label=r"$\zeta_{A_1}$",
                      style=dict(color='blue', marker='', linestyle='dashed')),
                 Prop("B->K^*::a2_uncertainty@BZ2004", 11, bandwidth=0.01,
                      legend_label=r"$\zeta_{A_2}$",
                      style=dict(color='green', linestyle='solid')),
                 Prop("B->K^*::v_uncertainty@BZ2004", 12,  bandwidth=0.01,
                      legend_label=r"$\zeta_{V}$",
                      style=dict(color=self.scen[scenario].c, marker='', linestyle='-.')))

        line_width = 1.5
        for p in props:
            p.style['linewidth'] = line_width

        prior_style = dict(color='black', linestyle='dotted', linewidth=line_width)
        prior_label = r"$\mathrm{prior}$"

        wide_figure()

        marg = self.margs[scenario]

        for i,p in enumerate(props):

            if p.bandwidth is not None:
                marg.use_histogram = False
                marg.kde_bandwidth = p.bandwidth
            else:
                marg.use_histogram = True

            # don't skip this plot
            marg.use_nuisance = True
            marg.use_contours = False
            marg.plot_prior = (i == 0)

            marg.one_dimensional(p.par_index, marginal_style=p.style, prior_style=prior_style,
                                 legend_label=p.legend_label, prior_label=prior_label)

        P.xlabel(r"$\zeta_i$")
        ax = P.gca()
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        P.setp(ax.get_yticklabels(), visible=False)

        # sort entries by label to push prior at the end
        handles, labels = ax.get_legend_handles_labels()
        import operator
        hl = sorted(zip(handles, labels), key=operator.itemgetter(1), reverse=True)
        handles2, labels2 = zip(*hl)
        ax.legend(handles2, labels2)
        P.legend(handles2, labels2, loc='upper left')

        P.savefig(self.out('nuis_1D_FF_BZ'))

    def single_scenario(self):
        """
        Plot one scenario by itself at a time.
        """

        P.figure(figsize=(8,8))
        tr = plotScript.Translator()
        for k in self.scen.keys():
            for i, p1 in enumerate(self.pars):
                for j, p2 in enumerate(self.pars[self.pars.index(p1) + 1:]):
                    # draw SM point
                    P.plot(self.sm_point[self.defs[p1].i], self.sm_point[self.defs[p2].i], **self.sm_point_style)

                    # draw best fit points
                    for n, point in enumerate(self.best_fit_points):
                        P.plot(point[self.defs[p1].i], point[self.defs[p2].i], \
                               **self.best_fit_points_style[n])

                    # add axis labels
                    P.xlabel(tr.to_tex(p1))
                    P.ylabel(tr.to_tex(p2))

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
        tr = plotScript.Translator()

        for i, p1 in enumerate(self.pars):
            for j, p2 in enumerate(self.pars[self.pars.index(p1) + 1:]):
                P.clf()
                # draw SM point
                P.plot(self.sm_point[self.defs[p1].i], self.sm_point[self.defs[p2].i], **self.sm_point_style)

                P.xlabel(tr.to_tex(p1))
                P.ylabel(tr.to_tex(p2))
                for k in self.scen.keys():
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
        plotScript.py sc1_all_nuis.hdf5 --single-1D 0 --contours --1D-bins 600 --use-data-range 0.05
        plotScript.py sc1_all_nuis.hdf5 --single-1D 0 --contours --1D-bins 250 --use-data-range 0.05 --use-KDE --bandwidth 0.005

        plotScript.py sc1_all_nuis.hdf5 --single-1D 1 --contours --1D-bins 300 --use-data-range 0.05
        plotScript.py sc1_all_nuis.hdf5 --single-1D 1 --contours --1D-bins 400 --use-data-range 0.05 --use-KDE --bandwidth 0.015

        plotScript.py sc1_all_nuis.hdf5 --single-1D 2 --contours --1D-bins 300 --use-data-range 0.05
        plotScript.py sc1_all_nuis.hdf5 --single-1D 2 --contours --1D-bins 400 --use-data-range 0.05 --use-KDE --bandwidth 0.015

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

        for s in ('all_nuis',):#'all_nuis',):,
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

if __name__ == '__main__':
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['font.size'] = 22
    matplotlib.rcParams['xtick.major.pad'] = 10
    matplotlib.rcParams['ytick.major.pad'] = 10
#    if os.environ['HOSTNAME'] == 'pcl128c':
#        matplotlib.rcParams['text.usetex'] = False

    # input_base = '/home/pcl312/beaujean/Data/EOS/bsll-paper/'
    # output_base = "/.th/pcl128c/scratch/beaujean/EOS/bsll-paper/"

    input_base = '/home/pcl312/beaujean/Data/EOS/proceeding/'
    output_base = "/.th/pcl128c/scratch/beaujean/EOS/proceeding/"

#    pull(output_base, mode=0, SM=False)

    marg = MarginalContours(input_base, output_base, max_samples=None,
                            ignore_scenarios=('all_wide',))#,'Vll_lowRec', 'Vll_largeRec', 'Pll'))
    # marg.one_dim_nuisance_KMPW()
    # marg.one_dim_nuisance_BZ()
    marg.single_scenario(); marg.overlays()
#    marg.all_nuis_vs_wide()
    marg.credibility_regions()

#    unc = UncertaintyMarginals(output_base, scenarios=['NP'], n_samples=None)
#    unc.plot()
#    unc.plot_large_recoil_single_bins()
#    unc.table()