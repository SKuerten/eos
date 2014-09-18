#! /usr/bin/env python
"""
Investigate the Amoroso and log gamma distribution.

Find parameter values from constraints on 68% [90/95% limits]
intervals numerically.

Ref: arxiv:1005.3274
"""
from __future__ import division
import math
import numpy as np
from numpy import log, exp
from scipy.special import gammaincc,gammainc, gamma, digamma
from scipy.optimize import fsolve
from scipy.stats.distributions import norm as Gaussian

import matplotlib
from matplotlib import rcParams
rcParams['text.usetex']= True #requires LaTex installation
rcParams['text.latex.unicode']=True
import pylab as P
import matplotlib.ticker

def mode(nu, lambdA, alpha):
    return nu + lambdA * log(alpha)

#logGamma
def pdf(x, nu, lambdA, alpha):
    if alpha <= 0:
        return 0

    z = (x - nu) / lambdA
    return 1 / gamma(alpha) / np.abs(lambdA) * exp(alpha * z - exp(z))

def cdf(x, nu, lambdA, alpha):
    if alpha <= 0:
        return 0

    z = (x - nu) / lambdA
    if lambdA < 0:
        return gammaincc(alpha, np.exp(z))
    else:
        return 1 - gammaincc(alpha, np.exp(z))

def cdf_limit(x, nu, lambdA, alpha):
    mu =  mode(nu, lambdA, alpha)
    if x < mu:
        return 0

    unnormalized_cdf_at_mode = cdf(mu, nu, lambdA, alpha)
#    print("unnorm dcdf at mode = %g " % unnormalized_cdf_at_mode)

    return (cdf(x, nu, lambdA, alpha) - unnormalized_cdf_at_mode) / (1.0 - unnormalized_cdf_at_mode)

def pdf_limit(x, nu, lambdA, alpha):
    mu =  mode(nu, lambdA, alpha)
    if x < mu:
        return 0

    y = (x - nu) / lambdA
    return 1.0 / (1.0 - cdf(mu, nu, lambdA, alpha))  / (gamma(alpha) * np.abs(lambdA)) * exp(alpha * y - exp(y))

def pdf_amoroso(x, a, theta, alpha, beta):
    z = (x - a) / theta
    return 1.0 / gamma(alpha) * np.abs(beta / theta) * z**(alpha * beta -1.0) * exp(-z**beta)

def cdf_amoroso(x, a, theta, alpha, beta):
    z =  (x - a) / theta
    if beta/theta < 0:
        return gammaincc(alpha, z**beta)
    else:
        return 1 - gammaincc(alpha, z**beta)

def constraints_amoroso_boundary(parameters, mode, x_90, x_95):
    """
    Two constraints for theta, alpha when fixing the mode
    at the boundary
    """
    theta = parameters[0]
    alpha = parameters[1]
    beta = 1.0 / alpha
    #a = mode

    #prevent optimization from getting stuck in invalid param. region, where result is NaN
    if alpha <= 0:
        return [1.0, 1.0]

#    print("theta = %g, alpha = %g" % (theta, alpha))

    first =  cdf_amoroso(x_90, mode, theta, alpha, beta) - 0.90
    second = cdf_amoroso(x_95, mode, theta, alpha, beta) - 0.95

#    print("first = %g, second = %g" % (first, second))

    if math.isnan(first):
        first = 0.5
    if math.isnan(second):
        second = 0.5

    return [first, second]

def constraints_amoroso_2012(parameters, a, x_10, x_50, x_90):
    """
    Three constraints for theta, alpha, beta from
    the 10%, 50% and 90% limit.
    """

    theta = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]
    #a = mode

    #prevent optimization from getting stuck in invalid param. region, where result is NaN
    if alpha <= 0:
        return [1.0, 1.0, 1.0]

#    print("theta = %g, alpha = %g" % (theta, alpha))

    third =  cdf_amoroso(x_10, a, theta, alpha, beta) - 0.10
    first =  cdf_amoroso(x_50, a, theta, alpha, beta) - 0.50
    second = cdf_amoroso(x_90, a, theta, alpha, beta) - 0.90
#    print("first = %g, second = %g" % (first, second))

    if math.isnan(first):
        first = 0.5
    if math.isnan(second):
        second = 0.5
    if math.isnan(third):
        third = 0.5

    return [first, second, third]

def constraints_amoroso(parameters, a, x_05, x_90, x_95):
    """
    Three constraints for theta, alpha, beta from
    the 05%, 90% and 95% limit.
    """

    theta = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]
    #a = mode

    #prevent optimization from getting stuck in invalid param. region, where result is NaN
    if alpha <= 0:
        return [1.0, 1.0, 1.0]

#    print("theta = %g, alpha = %g" % (theta, alpha))

    first =  cdf_amoroso(x_90, a, theta, alpha, beta) - 0.90
    second = cdf_amoroso(x_95, a, theta, alpha, beta) - 0.95
    third =  cdf_amoroso(x_05, a, theta, alpha, beta) - 0.05
#    print("first = %g, second = %g" % (first, second))

    if math.isnan(first):
        first = 0.5
    if math.isnan(second):
        second = 0.5
    if math.isnan(third):
        third = 0.5

    return [first, second, third]

def constraints_amoroso_mode(parameters, a, mode, x_90, x_95):
    """
    Two constraints for theta, alpha, beta from
    the mode, the 90% and 95% limit.
    For use with LHCb limit on B_s->mu mu, where
    we know the mode as well.
    """

    alpha = parameters[0]
    beta = parameters[1]
    theta = mode * (alpha - 1.0 / beta)**(-1.0 / beta)

    #prevent optimization from getting stuck in invalid param. region, where result is NaN
    if alpha <= 0:
        return [1.0, 1.0, 1.0]

#    print("theta = %g, alpha = %g" % (theta, alpha))

    first =  cdf_amoroso(x_90, a, theta, alpha, beta) - 0.90
    second = cdf_amoroso(x_95, a, theta, alpha, beta) - 0.95
#    print("first = %g, second = %g" % (first, second))

    if math.isnan(first):
        first = 0.5
    if math.isnan(second):
        second = 0.5

    return [first, second]

def constraints_amoroso_mode68(parameters, a, mode, sigma_plus, sigma_minus):
    """
    Use mode, 68% interval [mu - sigma_minus, mu + sigma_plus], and pdf(mu - sigma_minus) = pdf(mu + sigma_plus)
    Developed for LHCb and CMS July, 2013
    """

    alpha = parameters[0]
    beta = parameters[1]
    theta = mode * (alpha - 1.0 / beta) ** (-1.0 / beta)

    # prevent optimization from getting stuck in invalid param. region, where result is NaN
    if alpha <= 0:
        return [1.0, 1.0]

    first = cdf_amoroso(mode + sigma_plus, a, theta, alpha, beta) - \
            cdf_amoroso(mode - sigma_minus, a, theta, alpha, beta) - 0.680220861154
    second = pdf_amoroso(mode + sigma_plus, a, theta, alpha, beta) - \
             pdf_amoroso(mode - sigma_minus, a, theta, alpha, beta)

    if math.isnan(first):
        first = 0.5
    if math.isnan(second):
        second = 0.5

    return [first, second]

def solve_amoroso(a, x_05, x_90, x_95, initial_guess):
#    initial_guess = np.array([x_90_initial, x_95_initial])

    (x, infodict, ier, mesg) = fsolve(constraints_amoroso, initial_guess, args=(a, x_05, x_90, x_95), full_output=True, maxfev=1500)
    print(mesg)

    # read out solutions
    theta = x[0]
    alpha = x[1]
    beta =  x[2]

    print("The solution from minpack: theta = %.10e, alpha = %.10e, beta = %.10e" % (theta, alpha, beta))
    print(infodict)

    print("Checking the solution: do they satisfy the constraints?")
    print("cdf_limit_05 = %e " % cdf_amoroso(x_05, a, theta, alpha, beta))
    print("cdf_limit_90 = %e " % cdf_amoroso(x_90, a, theta, alpha, beta))
    print("cdf_limit_95 = %e " % cdf_amoroso(x_95, a, theta, alpha, beta))

    mode = a + theta * (alpha - 1 / beta)**(1 / beta)
    print("Mode at %e" % mode)

    print("cdf_limit_25 = %e " % cdf_amoroso(0.26031739193, a, theta, alpha, beta))
    print("cdf_limit_75 = %e " % cdf_amoroso(0.680220861154, a, theta, alpha, beta))

    print(log(pdf_amoroso(0.53, a, theta, alpha, beta)))
    print(log(pdf_amoroso(0.077590885823, a, theta, alpha, beta)))

    x_max = a + 2 * x_95
    x = np.linspace(a, x_max, 1000)
    y1 = [pdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]
    y2 = [cdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]

    P.plot(x,y1,label='pdf')
    P.plot(x,y2,label='cdf')

#    P.axhline(0.05, 0, 1, label='05\%', linestyle='--', color='black')
#    P.axhline(0.90, 0, 1, label='90\%', linestyle='--', color='black')
#    P.axhline(0.95, 0, 1, label='95\%', linestyle='--', color='black')
    P.axhline(0.05, 0, 1, linestyle='--', color='black')
    P.axhline(0.90, 0, 1, linestyle='--', color='black')
    P.axhline(0.95, 0, 1, linestyle='--', color='black')

    data = np.loadtxt('LHCb/Beaujean_Bayes.dat')
    br = data.T[0]
    cdf = data.T[1]
    P.plot(br, cdf,label=r"$\mbox{LHCb cdf}$")

    P.legend()
    P.title(r"$\theta = " + ("%g" % theta) + r", \alpha = " + ("%g" % alpha) + r", \beta = " + ("%g" % beta) + "$\n"
            + "mode at $" + "%g" % mode + "$")

    ax = P.gca()
    P.xlim(a, 1.8)
    P.ylim(0, 1.2)
#    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
#    P.xticks(np.linspace(a, a + 2 * x_95, 10))
#    P.grid()
    P.xlabel(r"BR$(B_s \to \mu \mu)$")
    P.savefig("B_s_limit.pdf")
    P.show()

# solve_amoroso(0.0, 0.077590885823, 0.932149816388, 1.10240011142, [0.6, 0.5, 2])

def solve_amoroso_mode(a, mode, x_90, x_95, initial_guess):
    """
    Fit the Amoroso distribution to info from LHCb 2012 Moriond talk on B_s -> mu mu
    """

    (x, infodict, ier, mesg) = fsolve(constraints_amoroso_mode, initial_guess, args=(a, mode, x_90, x_95), full_output=True, maxfev=1500)
    print(mesg)

    # read out solutions
    alpha = x[0]
    beta =  x[1]
    theta = mode * (alpha - 1.0 / beta)**(-1.0 / beta)
    assert(alpha * beta > 1)

    print("The solution from minpack: theta = %.10e, alpha = %.10e, beta = %.10e" % (theta, alpha, beta))
    print(infodict)

    print("Checking the solution: do they satisfy the constraints?")
    print("cdf_limit_90 = %e " % cdf_amoroso(x_90, a, theta, alpha, beta))
    print("cdf_limit_95 = %e " % cdf_amoroso(x_95, a, theta, alpha, beta))

    print(log(pdf_amoroso(0.53, a, theta, alpha, beta)))
    print(log(pdf_amoroso(0.077590885823, a, theta, alpha, beta)))

    x_max = a + 2 * x_95
    x = np.linspace(a, x_max, 1000)
    y1 = [pdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]
    y2 = [cdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]

    P.plot(x,y1,label='LHCb 2012 Moriond pdf')
    P.plot(x,y2,label='LHCb 2012 Moriond cdf')

#    P.axhline(0.05, 0, 1, label='05\%', linestyle='--', color='black')
#    P.axhline(0.90, 0, 1, label='90\%', linestyle='--', color='black')
#    P.axhline(0.95, 0, 1, label='95\%', linestyle='--', color='black')
    P.axhline(0.05, 0, 1, linestyle='--', color='black')
    P.axhline(0.90, 0, 1, linestyle='--', color='black')
    P.axhline(0.95, 0, 1, linestyle='--', color='black')

    data = np.loadtxt('LHCb/Beaujean_Bayes_2012_Moriond.dat')
    br = data.T[0] / 10.0
    cdf = data.T[1]
    P.plot(br, cdf,label=r"$\mbox{LHCb Heinrich 2012 cdf}$")

    P.legend()
    P.title(r"$\theta = " + ("%g" % theta) + r", \alpha = " + ("%g" % alpha) + r", \beta = " + ("%g" % beta) + "$\n"
            + "mode at $" + "%g" % mode + "$")

    ax = P.gca()
    P.xlim(a, 1.4)
    P.ylim(0, 4)
#    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
#    P.xticks(np.linspace(a, a + 2 * x_95, 10))
#    P.grid()
    P.xlabel(r"BR$(B_s \to \mu \mu)$")
    P.savefig("B_s_limit_2012.pdf")
    P.show()

def solve_amoroso_2012(a, x_10, x_50, x_90, initial_guess):
    """
    Interpolate results from LHCb Moriond 2012
    with 10, 50, 90 % limits.
    """
    (x, infodict, ier, mesg) = fsolve(constraints_amoroso_2012, initial_guess, args=(a, x_10, x_50, x_90), full_output=True, maxfev=1500)
    print(mesg)

    # values obtainted when fixing mode to 0.8 and using CL_S limits for 90% and 95%
#    x = (3.0196974712e-0, 5.9770060956e-01, 1.9231744175)

    # read out solutions
    theta = x[0]
    alpha = x[1]
    beta =  x[2]
    assert(alpha * beta > 1)

    print("The solution from minpack: theta = %.10e, alpha = %.10e, beta = %.10e" % (theta, alpha, beta))
    print(infodict)

    print("Checking the solution: do they satisfy the constraints?")
    print("cdf_limit_10 = %e " % cdf_amoroso(x_10, a, theta, alpha, beta))
    print("cdf_limit_50 = %e " % cdf_amoroso(x_50, a, theta, alpha, beta))
    print("cdf_limit_90 = %e " % cdf_amoroso(x_90, a, theta, alpha, beta))

    mode = a + theta * (alpha - 1 / beta)**(1 / beta)
    print("Mode at P(%e) = %.15e" % (mode, pdf_amoroso(mode, a, theta, alpha, beta)))

    p = cdf_amoroso(4.016344136, a, theta, alpha, beta) - cdf_amoroso(0.062156598126926255, a, theta, alpha, beta)
    print("checking significance: %g vs %g gives p of %g and pull of %g" % (pdf_amoroso(0.062156598126926255, a, theta, alpha, beta),
                                                                         pdf_amoroso(4.016344136, a, theta, alpha, beta),
                                                                         p, Gaussian.ppf( (p + 1) / 2)))
    p = cdf_amoroso(2.3050479788645215, a, theta, alpha, beta) - cdf_amoroso(0.516344136, a, theta, alpha, beta)
    print("checking significance: %g vs %g gives p of %g and pull of %g" % (pdf_amoroso(2.3050479788645215, a, theta, alpha, beta),
                                                                         pdf_amoroso(0.516344136, a, theta, alpha, beta),
                                                                         p, Gaussian.ppf( (p + 1) / 2)))

    ###
    # Compare the accuracy of interpolation
    ##
    data = np.loadtxt('LHCb/Beaujean_Bayes_2012_Moriond.dat')
    interpolated_cdf = np.array([cdf_amoroso(x_i, a, theta, alpha, beta) for x_i in data.T[0]])

    i = np.argmax(interpolated_cdf - data.T[1])
    print("Biggest deviation at %g: %g rel: %g" % (data[i,0], interpolated_cdf[i] - data[i,1], \
                                                   (interpolated_cdf[i] - data[i,1]) / data[i,1]))
    ###
    # Plot
    ##

    x_max = a + 2 * x_90
    x = np.linspace(a, x_max, 1000)
    y1 = [pdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]
    y2 = [cdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]

    P.plot(x,y1,label='pdf')
    P.plot(x,y2,label='cdf')

#    P.axhline(0.05, 0, 1, label='05\%', linestyle='--', color='black')
#    P.axhline(0.90, 0, 1, label='90\%', linestyle='--', color='black')
#    P.axhline(0.95, 0, 1, label='95\%', linestyle='--', color='black')
    P.axhline(0.10, 0, 1, linestyle='--', color='black')
    P.axhline(0.50, 0, 1, linestyle='--', color='black')
    P.axhline(0.90, 0, 1, linestyle='--', color='black')

    br = data.T[0]
    cdf = data.T[1]
    P.plot(br, cdf,label=r"$\mbox{LHCb Heinrich Moriond 2012 cdf}$")

    P.legend(ncol=3)
    P.title(r"$\theta = " + ("%g" % theta) + r", \alpha = " + ("%g" % alpha) + r", \beta = " + ("%g" % beta) + "$\n"
            + "mode at $" + "%g" % mode + "$")

    ax = P.gca()
    P.xlim(a, 6.5)
    P.ylim(0, 1.2)

    #2011
    P.xlim(a, 20)
    P.ylim(0, 1.2)
#    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
#    P.xticks(np.linspace(a, a + 2 * x_90, 10))
#    P.grid()
    P.xlabel(r"BR$(B_s \to \mu \mu)$")
    P.savefig("B_s_limit_2011.pdf")
    P.show()
#solve_amoroso_2012(0.0, 0.558367940293, 2.03115589965, 4.4528950788,[3.0, 1, 1.5])

def solve_amoroso_mode68(a, mode, sigma_plus, sigma_minus, initial_guess, experiment):
    """
    Fit the Amoroso distribution to info from LHCb and CMS July 2013 results

    :param a:

        minimum value, PDF=0 for x <= a. Typically a = 0.

    :param mode:

        position of most likely value reported by experiment

    :param sigma_plus, sigma_minus:

        uncertainty for x >[<] mode

    :param initial_guess:

        Initial guess for the parameters alpha and beta.

    :param experiment:

        String; used for labels and file name of plot.
    """
    print(constraints_amoroso_mode68(initial_guess, a, mode, sigma_plus, sigma_minus))
    (x, infodict, ier, mesg) = fsolve(constraints_amoroso_mode68, initial_guess,
                                      args=(a, mode, sigma_plus, sigma_minus), full_output=True, maxfev=1500)
    print(mesg)

    # read out solutions
    alpha = x[0]
    beta = x[1]
    theta = mode * (alpha - 1.0 / beta) ** (-1.0 / beta)
    assert(alpha * beta > 1)

    print("The solution from minpack: theta = %g, alpha = %g, beta = %g" % (theta, alpha, beta))
    print(infodict)

    print("Checking the solution: do they satisfy the constraints?")
    print("pdf(mu-sigma-) = %e " % pdf_amoroso(mode - sigma_minus, a, theta, alpha, beta))
    print("pdf(mu+sigma+) = %e " % pdf_amoroso(mode + sigma_plus, a, theta, alpha, beta))
    print("prob in %s = %e" % ((mode - sigma_minus, mode + sigma_plus),
                               cdf_amoroso(mode + sigma_plus, a, theta, alpha, beta) -
                               cdf_amoroso(mode - sigma_minus, a, theta, alpha, beta),))

    x_max = 3 * mode
    x = np.linspace(a, x_max, 1000)
    y1 = [pdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]
    y2 = [cdf_amoroso(x_i, a, theta, alpha, beta) for x_i in x]

    P.plot(x, y1, label='%s pdf' % experiment)
    P.plot(x, y2, label='%s cdf' % experiment)

    P.legend(loc='center right')
    P.title(r"$\theta = " + ("%g" % theta) + r", \alpha = " + ("%g" % alpha) + r", \beta = " + ("%g" % beta) + "$\n"
            + "mode at $" + "%g" % mode + "$")
    P.axhline(0.95, 0, 1, linestyle='--', color='black')

    ax = P.gca()
#     P.xlim(a, 1)

#    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
#    P.xticks(np.linspace(a, a + 2 * x_95, 10))
#    P.grid()
    P.xlabel(r"BR$(B_s \to \mu \mu)$")
    P.savefig("%s.pdf" % experiment)
#     P.show()

def plot_around():

    mode = 0.0
    x_90 = 6
    x_95 = 10

    lambdA = 1.47
    alpha = 12.1
    nu = mode - lambdA * np.log(alpha)

    print("cdf_limit_90 = %g " % cdf_limit(x_90, nu, lambdA, alpha))
    print("cdf_limit_95 = %g " % cdf_limit(x_95, nu, lambdA, alpha))

    x = np.linspace(-0.1, 1, 50)
    y = [cdf_limit(x_i, nu, lambdA, alpha) for x_i in x]

    P.plot(x,y)
    P.show()

def constraints2(parameters, mode, a, b):
    """
    Constraints: f(a) = f(b), CDF(b) - CDF(a) = 0.68, mode = mu
    """
    lambdA = parameters[0]
    alpha = parameters[1]
    nu = mode - lambdA * log(alpha)

    first = pdf(b, nu, lambdA, alpha) - pdf(a, nu, lambdA, alpha)
    second = (cdf(b, nu, lambdA, alpha) - cdf(a, nu, lambdA, alpha)) - 0.68268949213708585

#    print("first = %g, second = %g, cdf(a) = %g " % (first, second, cdf(a, nu, lambdA, alpha)))

    return [first, second]

def plot_log_gamma():
    mode = 0.53
    lambdA = 0.21
    alpha = 2.55
    nu = mode - lambdA * log(alpha)

    x = np.linspace(0, 1, 100)
    y1 = [pdf(x_i, nu, lambdA, alpha) for x_i in x]
    y2 = [cdf(x_i, nu, lambdA, alpha) for x_i in x]

    P.plot(x, y1)
    P.show()

def plot_log_gamma_starting_values():

    # sigma_+, -lambda, alpha
    values = np.array([
                       [1.03, 12.4, 127],
                       [1.04, 8.7, 72.4],
                       [1.1, 3.67, 12.4],
                       [1.3, 1.44, 1.78],
                       [1.6, 0.88, 0.64],
                       [1.8, 0.73, 0.44],
                       [1.9, 0.69, 0.38],
                       [2.0, 0.65, 0.33],
                       [2.5, 0.53, 0.21],
                       ])
    #transform to actual lambda
    values.T[1] *= -1.0

    print(values)

    # lambda
#    P.plot(values.T[0], values.T[1])
#    lambda_initial = [-56. + 55 * Gaussian.cdf(s-1.0, scale = 0.05) for s in values.T[0]]
#    P.plot(values.T[0], lambda_initial)
#    print(lambda_initial)
#    print(lambda_initial - values.T[1])
#    print(-65.5 + 65 * Gaussian.cdf(1.9,location=1, scale = 0.7))
#    print(Gaussian.cdf(0.9,location=0, scale = 0.7))
#    print(Gaussian.cdf(1.9,location=1, scale = 0.7))

    #alpha
    P.yscale('log')
    P.plot(values.T[0], values.T[2])
    alpha_initial = [(1.08 / (s - 1))**1.15 for s in values.T[0]]
    print(alpha_initial)
    print(alpha_initial - values.T[2])
    P.plot(values.T[0], alpha_initial)

    from scipy import interpolate

    finer_mesh = np.linspace(values.T[0,0], values.T[0,-1], 100)

    tck = interpolate.splrep(values.T[0], values.T[2], s=0, k=1)
    densities_interp = interpolate.splev(finer_mesh, tck, der=0)

    print(values.T[1])

#    interp_function = interpolate.interp1d(values.T[0], values.T[2], kind='linear')

    P.plot(finer_mesh, densities_interp)

#    P.xlim(1.1, 2)

    P.show()

#plot_log_gamma_starting_values()

def solve_log_gamma2(mode, a, b, lambda_initial, alpha_initial):
    """
    Constraints: f(a) = f(b), CDF(b) - CDF(a) = 0.68, mode = mu
    """
    initial_guess = np.array([lambda_initial, alpha_initial])


    (x, infodict, ier, mesg) = fsolve(constraints2, initial_guess, args=(mode, a, b), full_output=True, maxfev=1500)
    print(mesg)

    # read out solutions
    lambdA = x[0]
    alpha = x[1]
    nu = mode - lambdA * log(alpha)


    print("The solution from minpack: lambdA = %.10e, alpha = %.10e" % (lambdA, alpha))
    print(infodict)

    print("")
    print("Checking the solution: do they satisfy the constraints?")
    print("pdf(a) = %e, pdf(b) = %e " % (pdf(a, nu, lambdA, alpha), pdf(b, nu, lambdA, alpha)) )
    print("P in [a,b] = %g" % (cdf(b, nu, lambdA, alpha) - cdf(a, nu, lambdA, alpha)))

    print("")
    print("Extra information:")
    print(" a = %g, mode = %G, b = %g" % ( a, mode, b))
    print("pdf(a) = %g, pdf(mode) = %g, pdf(b) = %g" % (pdf(a, nu, lambdA, alpha), pdf(mode, nu, lambdA, alpha), pdf(b, nu, lambdA, alpha)))
    print("cdf(a) = %g, cdf(mode) = %g, cdf(b) = %g" % (cdf(a, nu, lambdA, alpha), cdf(mode, nu, lambdA, alpha), cdf(b, nu, lambdA, alpha)))

    x = np.linspace(a - 2 * (mode - a), b +  2 * (b - mode), 100)
    y1 = [pdf(x_i, nu, lambdA, alpha) for x_i in x]
    y2 = [cdf(x_i, mode, lambdA, alpha) for x_i in x]

    P.plot(x,y1,label='pdf')
#    P.plot(x,y2,label='cdf')
    constant_line = [pdf(a, nu, lambdA, alpha) for x_i in x]
    vertical_line_a = [(a, 0.0), (a, pdf(a, nu, lambdA, alpha))]
    P.plot(x, constant_line, label=r'pdf($1 \sigma$)', color='black')
    P.axvline(a, color='black')
    P.axvline(b, color='black')
    P.legend()
    P.title(r"$\nu = " + ("%.5g" % nu) + r", \lambda = " + ("%.5g" % lambdA) + r", \alpha = " + ("%.5g" % alpha) + " $")
    P.grid()
#    P.show()

    return (nu, lambdA, alpha)

#sol = solve_log_gamma2(0.53, 0.53 - 0.19, 0.53 + 0.10, 0.21, 2.55)
#sol = solve_log_gamma2(0.34, 0.32, 0.39, -0.01, 0.22)
#sol = solve_log_gamma2(0.0, - 1, 10,  -0.5, 0.2)
#sol = solve_log_gamma2(1.2, 0.66, 1.8,  -3.54, 11.81)
#sol = solve_log_gamma2(0.0, - 1, 1.08,  -5, 6.69)
#print("Rescaled solution: lambda = %g" % 0.10 * sol[1])
#print(log(pdf(0.57, sol[0], sol[1], sol[2])))
#print(log(pdf(0.92, sol[0], sol[1], sol[2])))
#print("mean = %g " % (sol[0] + sol[1] * digamma(sol[2])))
#print("rescaled pdf = %g" % (log(1/(cdf(0.7, sol[0], sol[1], sol[2]) - cdf(0.2, sol[0], sol[1], sol[2])) * pdf(0.57, sol[0], sol[1], sol[2]))))
#print(cdf(0.7, sol[0], sol[1], sol[2]) - cdf(0.2, sol[0], sol[1], sol[2]))
#print(pdf(0, 2.0548929, -6.945688154, 1.344270643))

def constraints(parameters, mode, x_90, x_95):

    lambdA = parameters[0]
    alpha = parameters[1]
    nu = mode - lambdA * log(alpha)

    first = np.abs(cdf_limit(x_90, nu, lambdA, alpha) - 0.90)
    second = np.abs(cdf_limit(x_95, nu, lambdA, alpha) - 0.95)

    return [first, second]

def solve_log_gamma1():

#    (x, infodict, ier, mesg) = fsolve(quadratic, [1], full_output=True)
#    print(mesg)
#    print(x)
#    print(infodict)
#    return

    # find some good starting values by hand


#    mode = 0.0
#    x_90 = 0.6
#    x_95 = 0.8
#
#    lambdA = 1.47
#    alpha = 12.1
    mode = 0.0
    x_90 = 0.9
    x_95 = 1.0

#    lambdA = -10.2
#    alpha = 100.1
#    nu = mode - lambdA * log(alpha)
#
#    print("cdf_limit_90 = %g " % cdf_limit(x_90, nu, lambdA, alpha))
#    print("cdf_limit_95 = %g " % cdf_limit(x_95, nu, lambdA, alpha))

#    return

    initial_guess = np.array([-1.2, 3.5])

    (x, infodict, ier, mesg) = fsolve(constraints, initial_guess, args=(mode, x_90, x_95), full_output=True, maxfev=1500)
    print(mesg)
    print(x)
    print(infodict)

    # read out solutions
    lambdA = x[0]
    alpha = x[1]
    nu = mode - lambdA * log(alpha)

    print("cdf_limit_90 = %g " % cdf_limit(x_90, nu, lambdA, alpha))
    print("cdf_limit_95 = %g " % cdf_limit(x_95, nu, lambdA, alpha))


def plot_amoroso():
    x = np.linspace(0, 3, 100)
    theta = 0.2
    alpha = 0.8
    y1 = [pdf_amoroso(x_i, a=0.0, theta=theta, alpha=alpha, beta=1/alpha) for x_i in x]
    y2 = [cdf_amoroso(x_i, a=0.0, theta=theta, alpha=alpha, beta=1/alpha) for x_i in x]

#    P.plot(x,y1)
#    P.plot(x,y2)

    nineties = np.array([10, 12, 13, 13.5, 14])
    thetas = np.array([0.0377, 0.5, 0.86, 1.0, 1.07])
    alphas = np.array([2.07, 0.757, 0.35, 0.196, 0.071])
    P.plot(nineties, thetas)
    P.plot(nineties, alphas)

    P.show()

def solve_amoroso_boundary(mode, x_90, x_95, x_90_initial, x_95_initial):
    initial_guess = np.array([x_90_initial, x_95_initial])

    (x, infodict, ier, mesg) = fsolve(constraints_amoroso, initial_guess, args=(mode, x_90, x_95), full_output=True, maxfev=1500)
    print(mesg)

    # read out solutions
    theta = x[0]
    alpha = x[1]
    beta = 1.0 / alpha

    print("The solution from minpack: theta = %.10e, alpha = %.10e" % (theta, alpha))
    print(infodict)

    print("Checking the solution: do they satisfy the constraints?")
    print("cdf_limit_90 = %e " % cdf_amoroso(x_90, mode, theta, alpha, beta))
    print("cdf_limit_90 = %e " % cdf_amoroso(x_90, mode, theta, alpha, beta))
    print("cdf_limit_95 = %e " % cdf_amoroso(x_95, mode, theta, alpha, beta))


    x = np.linspace(mode, mode + 2 * x_95, 100)
    y1 = [pdf_amoroso(x_i, mode, theta, alpha, beta) for x_i in x]
    y2 = [cdf_amoroso(x_i, mode, theta, alpha, beta) for x_i in x]

    P.plot(x,y1,label='pdf')
    P.plot(x,y2,label='cdf')
    constant_line90 = [0.90 for x_i in x]
    constant_line95 = [0.95 for x_i in x]
    P.plot(x, constant_line90, label='90%')
    P.plot(x, constant_line95,label='95%')
    P.legend()
#    P.title(r"$\theta = " + str(theta) + r", \alpha = " + str(alpha) + "$")
    P.title(r"$x_{90} = " + str(x_90) + r", x_{95} = " + str(x_95) + r"$")
    ax = P.gca()
#    ax.xaxis.set_major_locator(MultipleLocator(0.1))
#    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
#    P.xticks(np.linspace(mode, mode + 2 * x_95, 10))
    P.grid()
    P.show()

def constraints3(parameters, nu, theta, alpha):
    a = parameters[0]
    b = parameters[1]

    first = pdf(b, nu, theta, alpha) - pdf(a, nu, theta, alpha)
    second = (cdf(b, nu, theta, alpha) - cdf(a, nu, theta, alpha)) - 0.68268949213708585

    return [first, second]

def solve_log_gamma3(nu, theta, alpha, a_initial, b_initial):
    """
    Fix parameters, find a and b.
    Constraints: f(a) = f(b), CDF(b) - CDF(a) = 0.68, mode = mu
    """
    initial_guess = np.array([a_initial, b_initial])

    (x, infodict, ier, mesg) = fsolve(constraints3, initial_guess, args=(nu, theta, alpha), full_output=True, maxfev=1500)
    print(mesg)
    print("a = %.15f \n b = %.15f" % (x[0], x[1]))


    return
    # manual optimization doesn't work
    """
    sigplus = 1.045
    sigminus = -1.47 * sigplus
    print( solve_log_gamma2(0, sigminus, sigplus, 1, 1))
    """

if __name__ == '__main__':
    pass
    # in units of 10^{-8}
    #checked several starting positions, only one (or no) mode is found
#    solve_amoroso_mode(0.0, 0.08, 0.38, 0.45, [0.6, 2])

    # 2012 data
#    solve_amoroso_2012(0.0, 0.558367940293, 2.03115589965, 4.4528950788,[3.0, 1, 1.5])
    # 2011 data
#    solve_amoroso_2012(0.0, 0.132749474699 * 10, 0.446663009589 * 10, 0.932149816388 * 10,[3.0, 1, 1.5])
#    sol = solve_log_gamma2(0.53, 0.53 - 0.19, 0.53 + 0.10, 0.21, 2.55)
#     solve_log_gamma3(nu=0, theta=1, alpha=1, a_initial=-1, b_initial=1)


    # B_s -> mu mu  2013
    # solve_amoroso_mode68(0.0, mode=3.0, sigma_plus=1.0, sigma_minus=0.9, initial_guess=[2.2682156277, 1.7296007586], experiment='CMS 1307.5025')
    # solve_amoroso_mode68(0.0, mode=2.9, sigma_plus=np.sqrt(1.1 ** 2 + 0.3 ** 2), sigma_minus=np.sqrt(1.0 ** 2 + 0.1 ** 2),
    #                     initial_guess=[2.2682156277, 1.7296007586], experiment='LHCb 1307.5024')
    solve_amoroso_mode68(0.0, mode=2.9, sigma_plus=0.7, sigma_minus=0.7,
                        initial_guess=[2.2682156277, 1.7296007586], experiment=r'$B_s \to \mu \mu$ CMS + LHCb 2013')
    P.figure()
    solve_amoroso_mode68(0.0, mode=3.6, sigma_plus=1.6, sigma_minus=1.4,
                        initial_guess=[2.2682156277, 1.7296007586], experiment=r'$B_d \to \mu \mu$ CMS + LHCb 2013')
    P.figure()
    solve_amoroso_mode68(0.0, mode=2.8, sigma_plus=0.7, sigma_minus=0.6,
                        initial_guess=[2.10578, 3.02417], experiment=r'$B_s \to \mu \mu$ CMS + LHCb 2014')
    P.figure()
    solve_amoroso_mode68(0.0, mode=3.9, sigma_plus=1.6, sigma_minus=1.4,
                        initial_guess=[2.2682156277, 1.7296007586], experiment=r'$B_d \to \mu \mu$ CMS + LHCb 2014')
