#!/usr/bin/python
# -*- coding: utf-8 -*-

import plotScript as marg

import commands
import matplotlib

import numpy as np
import os
import pylab as P

# defines order of experiments
experiments = ['CLEO', 'BaBar', 'Belle', 'CDF', 'LHCb']

class ObservableMatcher(object):
    """Translate observables to (LaTex) group names"""

    # define special cases due binning variants:
    # a) LHCb variant up to 19.00
    # b) BaBar from 14.21
    observables = {}
    observables['B^+->K^+mu^+mu^-::BR[14.18,16.00]'] = 'B^+->K^+mu^+mu^-::BR[14,16]'
    observables['B^+->K^+mu^+mu^-::BR[14.21,16.00]'] = observables['B^+->K^+mu^+mu^-::BR[14.18,16.00]']
    observables['B^+->K^+mu^+mu^-::BR[16.00,22.86]'] = 'B^+->K^+mu^+mu^-::BR[>16]'

    observables['B^0->K^*0mu^+mu^-::BR[14.18,16.00]'] = 'B^0->K^*0mu^+mu^-::BR[14,16]'
    observables['B^0->K^*0mu^+mu^-::BR[14.21,16.00]'] = observables['B^0->K^*0mu^+mu^-::BR[14.18,16.00]']
    observables['B^0->K^*0mu^+mu^-::BR[16.00,19.21]'] = 'B^0->K^*0mu^+mu^-::BR[>16]'
    observables['B^0->K^*0mu^+mu^-::BR[16.00,19.00]'] = observables['B^0->K^*0mu^+mu^-::BR[16.00,19.21]']
    observables['B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]'] = 'B^0->K^*0mu^+mu^-::A_FB[>16]'
    observables['B^0->K^*0mu^+mu^-::A_FB[16.00,19.00]'] = observables['B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]']
    observables['B^0->K^*0mu^+mu^-::F_L[16.00,19.21]'] = 'B^0->K^*0mu^+mu^-::F_L[>16]'
    observables['B^0->K^*0mu^+mu^-::F_L[16.00,19.00]'] = observables['B^0->K^*0mu^+mu^-::F_L[16.00,19.21]']
    observables['B^0->K^*0mu^+mu^-::S_3[16.00,19.21]'] = 'B^0->K^*0mu^+mu^-::S_3[>16]'
    observables['B^0->K^*0mu^+mu^-::S_3[16.00,19.00]'] = observables['B^0->K^*0mu^+mu^-::S_3[16.00,19.21]']
    observables['B^0->K^*0mu^+mu^-::A_T_2[16.00,19.21]'] = 'B^0->K^*0mu^+mu^-::A_T_2[>16]'
    observables['B^0->K^*0mu^+mu^-::A_T_2[16.00,19.00]'] = observables['B^0->K^*0mu^+mu^-::A_T_2[16.00,19.21]']

    groups = {
              'B^+->K^+mu^+mu^-::BR[14,16]':'\mathcal{B}[14,16]',
              'B^+->K^+mu^+mu^-::BR[>16]':'\mathcal{B}[>16]',
              'B^0->K^*0mu^+mu^-::BR[14,16]':'\mathcal{B}[14,16]',
              'B^0->K^*0mu^+mu^-::BR[>16]':'\mathcal{B}[>16]',
              'B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]':'A_{\mathrm{FB}}[14,16]',
              'B^0->K^*0mu^+mu^-::A_FB[>16]':'A_{\mathrm{FB}}[>16]',
              'B^0->K^*0mu^+mu^-::F_L[14.18,16.00]':'F_L[14,16]',
              'B^0->K^*0mu^+mu^-::F_L[>16]':'F_L[>16]',
              'B^0->K^*0mu^+mu^-::S_3[14.18,16.00]':'S_{3}[14,16]',
              'B^0->K^*0mu^+mu^-::S_3[>16]':'S_3[>16]',
              'B^0->K^*0mu^+mu^-::A_T_2[14.18,16.00]':'A_T^{(2)}[14,16]',
              'B^0->K^*0mu^+mu^-::A_T_2[>16]':'A_T^{(2)}[>16]',
             }
    @staticmethod
    def to_group(eos_name):
        if eos_name in ObservableMatcher.observables:
            return ObservableMatcher.observables[eos_name]
        else:
            return eos_name

    @staticmethod
    def to_tex(group_name):
        if group_name in ObservableMatcher.groups:
            return '$' + ObservableMatcher.groups[group_name] + '$'
        else:
            return marg.Translator.to_tex(group_name)

class ConstraintInfo(object):
    """
    Store plot settings for individual constraints
    """
    def __init__(self, group_name, experiment, sigma):
        self.experiment = experiment
        self.group_name = group_name
        self.sigma = sigma
        self.tex_name = ObservableMatcher.to_tex(group_name)

    def __cmp__(self, other):
        return cmp(experiments.index(self.experiment), experiments.index(other.experiment))

def make_constraint_names(obs, decay='B^0->K^*0mu^+mu^-'):
    """
    Create a list of constraint names

    Example: make_constraint_names('BR') returns ['B^0->K^*0mu^+mu^-::BR[1.00,6.00]','B^0->K^*0mu^+mu^-::BR[14.18,16.00]', 'B^0->K^*0mu^+mu^-::BR[>16]']

    """
    # for B -> K*
    kinematics = ['[1.00,6.00]', '[14.18,16.00]', '[>16]']
    if obs == 'BR':
        kinematics[1] = '[14,16]'
    if decay == 'B^0->K^*0gamma':
        kinematics = ['']
    return ['%s::%s%s' % (decay, obs, k) for k in kinematics]

class PullPlot(object):
    """
    Read and plot significances at best fit point
    """
    def __init__(self):
        # key: observable part of constraint, value: list[ exp1, exp2...]
        self.constraint_infos = {}

        # store plot style for each experiment separately
        self.exp_plot_options = {}
        self.exp_plot_options['BaBar'] = dict(color='green')
        self.exp_plot_options['Belle'] = dict(color='blue')
        self.exp_plot_options['CDF']   = dict(color='red')
        self.exp_plot_options['CLEO']  = dict(color='salmon')
        self.exp_plot_options['LHCb']  = dict(color='yellow')

        # keep group of observables
        self.observable_groups = {}
        self.observable_groups['B^0->K^*0mu^+mu^-::F_L[16.00,19.00]'] = '$F_L[>16]$'
        self.observable_groups['B^0->K^*0mu^+mu^-::F_L[16.00,19.21]'] = '$F_L[>16]$'

    def read_data(self, file_name):
        """
        Fill constraint_infos
        """
        import h5py

        f = h5py.File(file_name, 'r')

        significances = f['/data/significances'][0]
        constraints = f['/descriptions/constraints']

        for i, row in enumerate(constraints):
            s = row[0]
            # extract observable part
            pos = s.find('@')
            key = s[0:pos]

            # extract name of experiment
            pos_old = pos
            pos = s.find('-', pos_old)
            exp = s[pos_old + 1:pos]
            print((key,exp, significances[i]))

            # use group as key, so 19.00 and 19.21 are plotted together
            group_name = ObservableMatcher.to_group(key)

            # store the parsed constraint
            c = ConstraintInfo(group_name=group_name, experiment=exp, sigma=significances[i])
            if group_name in self.constraint_infos:
                self.constraint_infos[group_name].append(c)
            else:
                self.constraint_infos[group_name] = [c]

    def plot(self, constraints, sig_max=3, left_adjust=0.1, right_adjust=0.95, bottom_adjust=0.08, top_adjust=0.8, padding=0.1,
             x_size=10, y_size=8, n_col=2, legend=True, handle_length=None):
        """
        Plot horizontal bar chart.

        Keyword arguments:
        constraints -- list of strings like ['B^0->K^*0mu^+mu^-::BR[1.00,6.00]', 'B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]']
        sig_max -- range of significances
        height -- todo
        left_adjust -- larger values give more room for long observable names on y-axis
        top_adjust -- larger values give less room for legend on above bar chart
        padding -- each group of constraints occupies a height of 1 - padding in arbitrary units; increases room between groups
        x_size -- size of figure in x-direction
        y_size -- size of figure in y-direction
        n_col -- number of columns in legend
        handle_length -- size of colored patch in legend (fraction of point size)
        """
        fig = P.figure(figsize=(x_size, y_size))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=left_adjust)
        fig.subplots_adjust(right=right_adjust)
        fig.subplots_adjust(top=top_adjust)
        fig.subplots_adjust(bottom=bottom_adjust)

        # store bar plot(artist) and exp name to fill legend later
        experiments_legend = {}

        # store observable tex names to set y-labels
        y_labels = []

        # find constraint with max# of experiments for padding
        n_max = max([len(self.constraint_infos[c]) for c in constraints])
        height = (1 - padding) / n_max

        # loop over observables in reverse, so first constraint is on top
        for i, c in enumerate(reversed(constraints)):
            y_labels.append(self.constraint_infos[c][0].tex_name)

            # loop over experiments/individual constraints, sorted by experiment
            for j,  exp in enumerate(sorted(self.constraint_infos[c])):
                #center single bars by adding 'empty' bars
                y_pos = i + (padding + height * (n_max - len(self.constraint_infos[c]))) / 2.0 + j * height
                experiments_legend[exp.experiment] = P.barh(y_pos, exp.sigma, height,
                                                          **self.exp_plot_options[exp.experiment])
        # set y labels to center of each observable bar stack
        ax.set_yticks(np.arange(len(y_labels)) + 0.5)
        ax.set_yticklabels(y_labels)

        # grid only for x-ticks, but below bars
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)

        # need ticks to show labels, but don't want to see them
        for tick in ax.get_yticklines():
            tick.set_visible(False)

        # add legend
        if legend:
            # decorate-sort-undecorate to respect experiment order
            decorated = [(experiments.index(exp), marg.Translator.to_tex(exp), legend_patch) for exp, legend_patch in experiments_legend.iteritems()]
            decorated.sort()
            legend_labels = [label for j, label, patch in decorated]
            legend_patches = [patch for j, label, patch in decorated]
            P.legend(legend_patches, legend_labels, bbox_to_anchor=(0., 1.02, 1., .102), \
                     loc=3, ncol=n_col, mode="expand", borderaxespad=0., handlelength=handle_length)

        ax.set_xlim(-sig_max, sig_max)
        ax.set_xlabel("$\delta$")

        ax.set_ylim(0, len(constraints))

def factory(cmd_line=None):
    """
    Create the uncertainty propagation object from command line arguments
    """
    import argparse

    parser  = argparse.ArgumentParser(description='Plot marginal distributions of MCMC')
    parser.add_argument('i', metavar='input file',  help='HDF5 input file name')
    parser.add_argument('--ext',  help='File type', action='store', default='pdf')

    # defaults to sys.argv if None is passed in
    args = parser.parse_args(cmd_line)

    pull_plot = PullPlot()

    pull_plot.read_data(args.i)

    return pull_plot

if __name__ == '__main__':
    pull_plot = factory()
