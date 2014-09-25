#!/usr/bin/python
# -*- coding: utf-8 -*-

import plotScript as marg
import translator

import matplotlib
import matplotlib.ticker as ticker
import numpy as np
import os
import pylab as P
import sys

class UncertaintyPropagation(object):
    """ Plot distribution of observables"""

    def __init__(self, file_names, one_sigma=True, ignore_cuts=False):

        # the observables
        self.data = []

        # the corresponding parameter samples
        self.parameters = []

        # with their weights
        self.weights = []

        ###
        # meta data
        ###

        # how many significant digits on mode for tables, format: observable_index:int(precision)
        # defaults to 3
        self.mode_precision = {}
        self.observable_names = {}
        # user can force a range, format: index:(x_min, x_max)
        self.observable_ranges = {}
        self.kinematics = {}
        self.sm_predictions = {}
        self.parameter_names = {}
        self.par_ranges = {}

        # read in data
        if len(file_names) == 1:
            self.file_names = list(file_names)
        else:
            self.file_names = file_names

        # adjust data range to make values that differ only in the
        # 10th digit come out as a delta peak in the distribution
        self.enforce_delta = True

        # determine the one sigma interval for the first distribution
        # and include it on the plot
        self.one_sigma = one_sigma

        # todo rethink cuts, especially on A_FB, it may be negative
        # if observable is above the cut, i.e. an outlier, remove it from plot
        self.lower_cuts = {
#                     "B_q->ll::BR,s_min=0,s_max=0":1e-9,
                     "B->K^*ll::A_FB@LargeRecoil,s_min=2,s_max=4.3":0.0,
#                     "B->K^*ll::A_T^2@LargeRecoil,s_min=2,s_max=4.3":-0.035,
                     "B->K^*ll::H_T^1@LargeRecoil,s_min=2,s_max=4.3":0.0,
                     "B->K^*ll::H_T^2@LargeRecoil,s_min=2,s_max=4.3":-0.7,
                     "B->K^*ll::H_T^3@LargeRecoil,s_min=2,s_max=4.3":0,
                     "B->K^*ll::H_T^5@LargeRecoil,s_min=2,s_max=4.3":-0.01,
                     "B->K^*ll::A_T^2@LargeRecoil,s_min=1,s_max=6":-1,
#                     "B->K^*ll::A_FB@LargeRecoil,s_min=1,s_max=6":-0.0,
                     "B->K^*ll::H_T^1@LargeRecoil,s_min=1,s_max=6":0.1,
                     "B->K^*ll::H_T^2@LargeRecoil,s_min=1,s_max=6":-0.6,
                     "B->K^*ll::H_T^3@LargeRecoil,s_min=1,s_max=6":-0.3,
                     "B->K^*::V(s)/A_1(s),s_min=0,s_max=0":0,
                     "B->K^*::A_2(s)/A_1(s),s_min=0,s_max=0":0.
                    }

        self.upper_cuts = {
                     "B_q->ll::BR,s_min=0,s_max=0":4e-9,
                     "B->Kll::BR@LargeRecoil,s_min=2,s_max=4.3":2e-7,
                     "B->K^*ll::BR@LargeRecoil,s_min=2,s_max=4.3":5e-7,
                     "B->K^*ll::A_FB@LargeRecoil,s_min=2,s_max=4.3":0.6,
                     "B->K^*ll::F_L@LargeRecoil,s_min=2,s_max=4.3":1,
#                     "B->K^*ll::A_T^3@LargeRecoil,s_min=2,s_max=4.3":-0.018,
                     "B->K^*ll::A_T^4@LargeRecoil,s_min=2,s_max=4.3":1.5,
                     "B->K^*ll::A_T^im@LargeRecoil,s_min=2,s_max=4.3":0.007,
                     "B->K^*ll::H_T^1@LargeRecoil,s_min=2,s_max=4.3":0.9,
                     "B->K^*ll::H_T^2@LargeRecoil,s_min=2,s_max=4.3":0.1,
                     "B->K^*ll::H_T^3@LargeRecoil,s_min=2,s_max=4.3":1,
                     "B->K^*ll::H_T^4@LargeRecoil,s_min=2,s_max=4.3":0.01,
#                     "B->Kll::BR@LargeRecoil,s_min=1,s_max=6":4e-7,
                     "B->K^*ll::BR@LargeRecoil,s_min=1,s_max=6":1.5e-6,
                     "B->K^*ll::F_L@LargeRecoil,s_min=1,s_max=6":1,
                     "B->K^*ll::A_T^3@LargeRecoil,s_min=1,s_max=6":1.0,
                     "B->K^*ll::A_T^4@LargeRecoil,s_min=1,s_max=6":1.7,
                     "B->K^*ll::H_T^1@LargeRecoil,s_min=1,s_max=6":0.9,
                     "B->K^*ll::H_T^2@LargeRecoil,s_min=1,s_max=6":0.0,
                     "B->K^*ll::H_T^4@LargeRecoil,s_min=1,s_max=6":0.0,
                     "B->K^*::V(s)/A_1(s),s_min=0,s_max=0":3,
                     "B->K^*::A_2(s)/A_1(s),s_min=0,s_max=0":3.
                    }

        if ignore_cuts:
            self.lower_cuts = {}
            self.upper_cuts = {}

        # print 1sigma interval in title of marginal plots
        self.print_uncertainty = True

        # rescale by orders of magnitude
        self.scale_factors = {}

#        self.output, ext = os.path.splitext(file_names[0])
#        self.output += ".pdf"
        if len(file_names) > 1:
#            self.output_base = os.path.split(file_names[0])[0]
            self.output_base = "/home/heinz/workspace/paper-bayes/figures/uncert-prop-"
            self.output = os.path.join(self.output_base, "combined.pdf")
        else:
            self.output_base = os.path.splitext(file_names[0])[0]
            self.output = self.output_base +".pdf"

        #strings for the legends
        #/path/to/data/CKM.log produces 'CKM'
        self.legend_titles = []
        for file in file_names:
            self.legend_titles.append(os.path.splitext(os.path.basename(file))[0])

        self.single_ext = ".pdf"

        self.n_bins = 100

    def __determine_file_type(self):
        """
        Check whether pmc or ordinary
        """

        import h5py

        file_types = []

        for name in self.file_names:
            f = h5py.File(self.file_names[0])

            try:
                f['/data/weights']
                file_types.append('pmc')
            except KeyError:
                file_types.append('prior')

        return file_types

    #todo read only single files
    def read_data_prior(self, file_name, select=None):
        "Format since end of Oct 2011"

        import h5py

        hdf5_file = h5py.File(file_name, 'r')

        ###
        # extract meta information
        ###

        # sort by numerical value
        for o in np.sort(np.array(list(hdf5_file['/descriptions/observables']),dtype=int)):
            dset = hdf5_file['/descriptions/observables/%d' % o]
            self.observable_names[o] = dset.attrs['name']
            self.kinematics[o] = (dset[0][0], dset[0][1])
            self.sm_predictions[o] = dset.attrs['SM prediction']

#        for key in sorted(self.observable_names.keys()):
#            sys.stdout.write( "%s: %s, " % (key, self.observable_names[key]))
#        sys.stdout.write('\n')

        self.n_observables = len(self.observable_names)

        try:
            for o in np.sort(np.array(list(hdf5_file['/descriptions/parameters']),dtype=int)):
                dset = hdf5_file['/descriptions/parameters/%d' % o]
                self.parameter_names[o] = dset.attrs['name']
                self.par_ranges[o] = (dset[0][0], dset[0][1])

            self.n_parameters = len(self.parameter_names)
        except KeyError:
            self.n_parameters = 0

        ###
        # extract the samples, parameters are optional
        ###

        size = hdf5_file['/data/observables'].len()
        if select is None:
            select = (0, size)
        else:
            size = select[1] - select[0]

        parameters = None
        try:
            parameters = hdf5_file['/data/parameters'][select[0]:select[1]]
            print("Parameters have a shape of %s" % repr(parameters))
        except KeyError:
            pass
        self.parameters.append(parameters)

        self.data.append(hdf5_file['/data/observables'][select[0]:select[1]])

        self.weights.append(np.ones(size))

    def read_data_pmc(self, file_name, select=None):
        """
        Parse data from uncertainty propagation based on
        posterior samples from PMC
        """
        import h5py

        hdf5_file = h5py.File(file_name, 'r')

        ###
        # extract meta information
        ###

        # sort by numerical value
        for o in np.sort(np.array(list(hdf5_file['/descriptions/observables']),dtype=int)):
            dset = hdf5_file['/descriptions/observables/%d' % o]
            self.observable_names[o] = dset.attrs['name']
            # skip trivial kinematics
            if dset[0][0] != dset[0][1]:
                self.kinematics[o] = (dset[0][0], dset[0][1])
            self.sm_predictions[o] = dset.attrs['SM prediction']

        self.n_observables = len(self.observable_names)

        # parse those from posterior descriptions
        for i,p in enumerate(hdf5_file['/descriptions/parameters']):
            self.parameter_names[i] = p['name']
            self.par_ranges[i] = (p['min'], p['max'])

        print(self.parameter_names)
        print(self.par_ranges)

        self.n_parameters = len(self.parameter_names)

        ###
        # extract the samples
        ###

        parameters = None
        self.parameters.append(parameters)

        size = hdf5_file['/data/observables'].len()
        if select is None:
            select = (0, size)
        else:
            size = select[1] - select[0]

        self.data.append(hdf5_file['/data/observables'][select[0]:select[1]])

        self.weights.append(np.exp(hdf5_file['/data/weights'][select[0]:select[1]].T['weight']))

        hdf5_file.close()

    def read_data(self, select=(None, None)):
        """Generic method to read in data, determines file type"""
        file_types = self.__determine_file_type()
        for t,f in zip(file_types, self.file_names):
            if t is 'pmc':
                self.read_data_pmc(f, select)
            else:
                self.read_data_prior(f, select)

        print("Observables have a shape of %s" % repr(self.data[-1].shape))

    def plot_all(self, input_template=None, single_args=dict()):

        pdf_file = matplotlib.backends.backend_pdf.PdfPages(self.output)

        output_template = input_template

        #loop over observables
        for i in range(self.data[0].shape[1]):
            P.figure()
            output_template = self.single_plot(i, output_template=output_template, **single_args)[0]
            pdf_file.savefig()

        pdf_file.close()

        return output_template

    def parameter_correlations(self):
        """ Plot all marginal distributions of parameter samples for
        a given condition on the observable identified by its index
        """
        pdf_file = PdfPages(os.path.splitext(self.output)[0]+"_correlations.pdf")

        #find the outliers
        i = 3
#        filtered_data = self.data[0].T[i]
#        filter_values = (6.7944303209373962e-08, 6.798250062612962e-08)
#        filter = np.where(np.array(filtered_data >= filter_values[0], dtype='int') * np.array(filtered_data <= filter_values[1],dtype='int'))[0]

        i = 0
        filtered_data = self.parameters[0].T[i]
        filter_values = (0.8093, 0.8097)
        filter = np.where(np.array(filtered_data >= filter_values[0], dtype='int')
                          * np.array(filtered_data <= filter_values[1],dtype='int'))[0]

        #loop over parameters
        for i in range(self.n_parameters):
            P.figure()
            print("plot %d of %d" % (i, self.n_parameters))
            bins = 80

            #first plot the full set
            P.hist(self.parameters[0].T[i], bins=bins, range=self.par_ranges[i], alpha=0.3)

            #now the outlier
            P.hist(self.parameters[0].T[i][filter], bins=1)

#            P.xlabel("Par "+str(i))
            P.xlabel(self.parameter_names[i])

            pdf_file.savefig()
            P.close()

        #2-D
        for i in range(self.n_parameters):
            for j in range(i+1, self.n_parameters):
                continue
                print("plot (%d, %d) of %d" % (i, j, self.n_parameters * (self.n_parameters - 1) / 2))

                bins = 50
                H_full, xedges, yedges = histogram2d(self.parameters[0].T[i],
                                                self.parameters[0].T[j],
                                                bins=bins)
               #convert to standard display order
                H_full = flipud(fliplr(rot90(H_full,k=3)))

                P.imshow(H_full,
                     cmap=  P.get_cmap('jet'),
                     extent = (xedges[0], xedges[-1], yedges[0], yedges[-1]),
                     interpolation = 'nearest')


                #now scatter plot of the outliers
                P.scatter(self.parameters[0].T[i][filter],
                          self.parameters[0].T[j][filter],
                          color='w', marker='o', s=10)

                P.axis('tight')

                P.xlabel("Par "+str(i))
                P.ylabel("Par "+str(j))

                pdf_file.savefig()
                P.close()


        pdf_file.close()

    def single_plot(self, obs_index, data_index=0, credibility=True,
                    credibility_style=dict(color='blue', alpha=0.4),
                    histo_style=dict(color='black', linestyle='solid'),
                    n_samples=None, range=None, output_template=None,
                    n_ticks=None, y_labels=True, skip_SM_prediction=False):
        """
        Plot a single histogram for a given observable.

        args:
        obs_index index of the observable
        data Specify the data file, defaults to first file.

        n_samples If none, use only the first n samples
        """

        data = self.data[data_index]

#        x_min = min(data.T[obs_index])
#        x_max = max(data.T[obs_index])

        # mask points with zero weight straight away, so range is set correctly
        masked_obs = np.ma.MaskedArray(data.T[obs_index], copy=False, mask=(self.weights[data_index]== 0.0))
        n_cut = np.ma.count_masked(masked_obs)
        if n_cut > 0:
            print("Warning: %d values have zero weight" % n_cut)

        full_observable_name = self.observable_names[obs_index]
        try:
            full_observable_name += ",s_min=%g,s_max=%g" % self.kinematics[obs_index]
        except KeyError:
            pass

        print(full_observable_name)

        try:
            cut_value = self.lower_cuts[full_observable_name]
            masked_obs = np.ma.masked_less(data.T[obs_index], cut_value, copy=False)
            n_cut = np.ma.count_masked(masked_obs) - n_cut
            if n_cut > 0:
                print("Warning: Using cut ( %s > %g )" % (full_observable_name, cut_value))
                print("Detected %d outliers" % n_cut)
        except KeyError:
            pass

        try:
            cut_value = self.upper_cuts[full_observable_name]
            masked_obs = np.ma.masked_greater(masked_obs, cut_value, copy=False)
            n_cut = np.ma.count_masked(masked_obs) - n_cut
            if n_cut > 0:
                print("Warning: Using cut (%s < %g)" % (full_observable_name, cut_value))
                print("Detected %d outliers" % n_cut)
        except KeyError:
            pass

        if range:
            x_min, x_max = range
        elif obs_index in self.observable_ranges:
            x_min, x_max = self.observable_ranges[obs_index]
        else:
            x_min = min(masked_obs)
            x_max = max(masked_obs)

        # hist does ignore masked values, as opposed to the discussion at
        # http://stackoverflow.com/questions/3610040/how-to-create-the-histogram-of-an-array-with-masked-values-in-numpy

        weights = np.ma.MaskedArray(self.weights[data_index], copy=False)#, mask=masked_obs.mask)#.filled(0.0)
        outline_hist, normal_hist = marg.histOutline(masked_obs, weights=weights,
                                                     range=(x_min, x_max), bins=self.n_bins, normed=True)

        #use this for drawing
        bin_edges, hist = outline_hist

        #store the highest density
        maximum_density = max(hist)

        #find credibility region
        if credibility:
            #add the last bin's right edge assuming constant bin width
            bin_edges_normal = np.hstack((normal_hist[0], np.array(normal_hist[0][-1] + (normal_hist[0][1]-normal_hist[0][0]))) )

            intervals = []

            # fill the 1sigma region
            (mode, sigma_lower, sigma_upper), (ind_central, ind_lower, ind_upper) = marg.find_credibility_region(bin_edges_normal, normal_hist[1])
            # add plus to for right edge of right bin
            intervals.append((bin_edges_normal[ind_lower], bin_edges_normal[ind_upper + 1]))
#            intervals.append((mode - sigma_lower, mode + sigma_upper))
            # + 1 to reach next original bin, 2 * and + 1 as points are duplicated for plotting
            P.fill_between(bin_edges[2*ind_lower:2*(ind_upper + 1) + 1],
                           np.zeros(2*(ind_upper + 1 - ind_lower) + 1), hist[2*ind_lower:2*(ind_upper + 1) + 1],
                           **credibility_style)

            #find uncertainties in proper format, but only of largest of the three values
            # to deal with large relative uncertainty
            trinity = abs(np.array([mode, sigma_lower, sigma_upper]))
            val = np.max(trinity)
            exponent = int(np.floor(np.log10(val)))

            std_mode = mode * 10**(-exponent)
            std_upper = sigma_upper * 10**(-exponent)
            std_lower = sigma_lower * 10**(-exponent)
            if exponent <= -2:
                uncertainty_string = "(%.3g \\; ^{+%.2g} _{-%.2g} ) \\cdot 10^{%d}" % (std_mode, std_upper, std_lower, exponent)
            else:
                uncertainty_string = "%.3g \\; ^{+%.2g} _{-%.2g}" % (mode, sigma_upper, sigma_lower)

            # only want 2 sigma values
            (central2, two_sigma_lower, two_sigma_upper), indices = marg.find_credibility_region(bin_edges_normal, normal_hist[1], alpha=0.9544997)
            intervals.append((bin_edges_normal[indices[1]], bin_edges_normal[indices[2] + 1]))

        P.plot(bin_edges, hist, label=self.legend_titles[data_index], **histo_style)

        # draw SM prediction if available
        P.plot([self.sm_predictions[obs_index], self.sm_predictions[obs_index]], [0.0, maximum_density], color='red', linestyle='-' )

        ax = P.gca()
        ax.xaxis.major.formatter.set_powerlimits((-4,4))
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        if output_template:
            rescaled_values = [mode, sigma_upper, sigma_lower]
            if skip_SM_prediction:
                rescaled_values.extend((two_sigma_upper, two_sigma_lower))
            else:
                rescaled_values.append(self.sm_predictions[obs_index])

            rescaled_values = np.array(rescaled_values)

            if full_observable_name.find('::J_') > -1:
                rescaled_values *= 1.525e-12 / 6.5821188926e-25  # lifetime of B_d [s] / conversion s to GeV^-1

            # rescale if forced by user
            if self.observable_names[obs_index] in self.scale_factors:
                scale = self.scale_factors[self.observable_names[obs_index]]
                magnitude = "\\times 10^{%d}" % scale
                rescaled_values *= 10**(scale)
            else:
                magnitude = ""

            #special case: J7 is much much smaller
            if abs(rescaled_values[3]) < 1e-10:
                rescaled_values[3] = 0.0

            print("rescaled values: %s" % rescaled_values)
            if obs_index in self.mode_precision:
                prec = self.mode_precision[obs_index]
            else:
                prec = 2

            sigma_prec = 2

            precisions = [prec, sigma_prec, sigma_prec]#, prec]
            if skip_SM_prediction:
                precisions.extend((sigma_prec, sigma_prec))
            else:
                precisions.append(prec)

            rescaled_formats = ["%." + str(p) + ("f" if p == prec else "g") for p in precisions]
            """
            rescaled_formats = ['mode', 'sigma+', 'sigma-', 'SM central']
            rescaled_formats[0] = "%." + str(prec) + "f"
            rescaled_formats[1] = "%." + str(sigma_prec) + "g"
            rescaled_formats[2] = "%." + str(sigma_prec) + "g"
            rescaled_formats[3] = "%." + str(prec) + "f"
            """
            rescaled_strings = [form % value for form, value in zip(rescaled_formats, rescaled_values)]

            # check for length and add trailing zero if needed
            for i,s in enumerate(rescaled_strings):
                v = rescaled_values[i]
                if v != 0.0:
                    exponent = int(np.floor(np.log10(np.abs(v))))
                else:
                    exponent = 0
                if exponent < 0:
                    if exponent > -5:
                        # one after the '.'
                        dot_pos = s.find('.')

                        # digits after '.'
                        n_digits = len(s) - dot_pos

                        # number of zeros directly after '.'
                        n_leading_zeros = len(s[dot_pos:]) - len(s[dot_pos:].lstrip('0'))

                        # want to have
                        target = -exponent + precisions[i] - 1 - n_leading_zeros

                        n_missing_zeros = target - n_digits
                        # empty string added if nothing is missing
                        rescaled_strings[i] = s + '0' * n_missing_zeros
                    else:
                        rescaled_formats[i] = "%." + str(abs(exponent) - 1) + "f"
                        rescaled_strings[i] = rescaled_formats[i] % v
                if exponent > 0:
                    decimal_index = s.find('.')
                    # turn '7' into '7.0'
                    if decimal_index < 0:
                        n_missing_zeros = precisions[i] - len(s) - 1
                        if n_missing_zeros > 0:
                            rescaled_strings[i] = s + '.' + '0' * n_missing_zeros

            if skip_SM_prediction:
                table_row = "$%s \\; ^{+ %s} _{- %s} \\; ^{+ %s} _{- %s}$" % tuple(rescaled_strings)
            else:
                table_row = "$%s \\; ^{+ %s} _{- %s} \\; (%s)$" % tuple(rescaled_strings)

            # replace only once, if there are several fields of the same name, latex will show the error
            output_template = output_template.replace(full_observable_name, table_row, 1)
            output_template = output_template.replace(self.observable_names[obs_index] + "_MAGNITUDE", magnitude, 1)

            print("Table row: %s" % table_row)

        #add kinematic info if available
        kinematic_string = ""
        try:
            s_min, s_max = self.kinematics[obs_index]
            kinematic_string = "$[%g,%g]$" % (s_min, s_max)
        except KeyError:
            pass

        # workaround for
        # matplotlib.pyparsing.ParseFatalException: Subscript/superscript sequence is too long. Use braces { } to remove ambiguity.
        if matplotlib.rcParams['text.usetex']:
            uncertainty_string = '$' + uncertainty_string + '$'
        if self.print_uncertainty:
            P.title('$' + translator.EOS_Translator().to_tex(self.observable_names[obs_index]) + '$' + kinematic_string + "$) = $" + uncertainty_string)
        else:
            P.xlabel(self.tr.to_tex(self.observable_names[obs_index]) + kinematic_string)
        P.xlim(x_min, x_max)

        # fine tune axes
        ax = P.gca()
        if n_ticks:
            ax.xaxis.set_major_locator(ticker.LinearLocator(n_ticks))
        if not y_labels:
#            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.major.formatter.set_powerlimits((-500, 500)) # avoid power offset separately
            P.setp(ax.get_yticklabels(), visible=False)

        return (output_template, mode, intervals)

def manual_filter(unc):
    """Apply cuts to an instance of UncertaintyPropagation"""

    unc.lower_cuts[r"B->K^*ll::P'_4@LargeRecoil,s_min=2,s_max=4.3"] = 0.0
    unc.lower_cuts[r"B->K^*ll::P'_4@LargeRecoil,s_min=1,s_max=6"] = 0.0
    unc.upper_cuts[r"B->K^*ll::P'_5@LargeRecoil,s_min=2,s_max=4.3"] = 0.0
    unc.upper_cuts[r"B->K^*ll::P'_5@LargeRecoil,s_min=1,s_max=6"] = 0.0
    unc.upper_cuts[r"B->K^*ll::P'_6@LargeRecoil,s_min=2,s_max=4.3"] = 0.0
    unc.upper_cuts[r"B->K^*ll::P'_6@LargeRecoil,s_min=1,s_max=6"] = 0.0

    unc.upper_cuts[r"B->K^*ll::P'_5@LowRecoil,s_min=14.18,s_max=16"] = 0.0
    unc.upper_cuts[r"B->K^*ll::P'_5@LowRecoil,s_min=16,s_max=19"] = 0.0


    unc.lower_cuts[r"B->K^*ll::P'_6@LowRecoil,s_min=14.18,s_max=16"] = -0.01
    unc.upper_cuts[r"B->K^*ll::P'_6@LowRecoil,s_min=14.18,s_max=16"] = +0.01
    unc.lower_cuts[r"B->K^*ll::P'_6@LowRecoil,s_min=16,s_max=19"] = -0.003
    unc.upper_cuts[r"B->K^*ll::P'_6@LowRecoil,s_min=16,s_max=19"] = +0.003

    return unc

def factory(cmd_line=None):
    """
    Create the uncertainty propagation object from command line arguments
    """
    import argparse

    parser  = argparse.ArgumentParser(description='Plot marginal distributions of MCMC')
    parser.add_argument('i', metavar='input file',  help='HDF5 input file name')
    parser.add_argument('--1D-bins', dest='one_dim_bins', help="Use fixed number of bins for 1D marginal distributions",action='store')
    parser.add_argument('--ignore-cuts',  help='Apply cuts on range of observables to remove outliers', action='store_true')
    parser.add_argument('--obs', help="Plot a single observable. Can be specified multiple times", action='append')
    parser.add_argument('--no-unc',  help='Do not print 1sigma intervals in title of 1D distributions', action='store_false')
    parser.add_argument('--select', help="Select a range of samples from each chain", action='store',nargs=2)
    parser.add_argument('--table', help="Output uncertainties into table", action='store')
    parser.add_argument('--use-data-range', help="Determine the parameter par_ranges from data, instead of from definition in HDF5. ", action='store', default=0.0)

    args = parser.parse_args(cmd_line)

    ###
    # setup the object
    ###

    uncert = UncertaintyPropagation([args.i], one_sigma=True, ignore_cuts= args.ignore_cuts)
    if args.select:
        args.select = (int(args.select[0]), int(args.select[1]))

    uncert.read_data(select=args.select)

    if args.one_dim_bins is not None:
        uncert.n_bins = int(args.one_dim_bins)

    uncert.print_uncertainty = args.no_unc

    if cmd_line is not None:
        return uncert

    if args.table:
        f = open(args.table)
        template = f.read()
    else:
        template = None

    uncert = manual_filter(uncert)

    ###
    # now come the actions
    ###

    if args.obs:
        for o in args.obs:
            P.figure()
            uncert.single_plot(int(o))
            P.savefig(uncert.output_base + uncert.single_ext)
    # default: plot all observables
    else:
        output_table = uncert.plot_all(template)
        if output_table:
            f = open(args.output)
            f.write(output_table)

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    matplotlib.rcParams['text.usetex'] = True
    factory()
