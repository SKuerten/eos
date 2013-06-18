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
from scipy.special import gammaincc,gammainc, gamma
from scipy.optimize import fsolve
import pylab as P
from matplotlib.ticker import MultipleLocator

def mode(nu, lambdA, alpha):
    return nu + lambdA * log(alpha)    

def cdf(x, nu, lambdA, alpha):
    """LogGamma"""
    if lambdA < 0:
        return gammaincc(alpha, np.exp((x - nu) / lambdA))
    else:
        return 1 - gammaincc(alpha, np.exp((x - nu) / lambdA))

def pdf(x, nu, lambdA, alpha):
    """LogGamma"""
    y = (x - nu) / lambdA
    return 1.0 / gamma(alpha) / np.abs(lambdA) * exp(alpha * y - exp(y))

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
    norm = 1.0 / gamma(alpha) * np.abs(beta / theta)
#    print("z^beta = %g, norm = %g" % (z**beta, np.log(norm)))
    return 1.0 / gamma(alpha) * np.abs(beta / theta) * z**(alpha * beta -1.0) * exp(-z**beta)

def cdf_amoroso(x, a, theta, alpha, beta):
    z =  (x - a) / theta
    if beta/theta < 0:
        return gammaincc(alpha, z**beta)
    else:
        return 1 - gammaincc(alpha, z**beta)
    
def constraints_amoroso(parameters, mode, x_90, x_95):
    
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
    
def plot_around():
    
    lambdA = 0.1046632085
    alpha = 1.921694426
    nu = 1.201633227

    x = np.linspace(0.5, 1.5, 100)
    y = [pdf(x_i, nu, lambdA, alpha) for x_i in x]
    
    P.plot(x,y)
    P.show()

def constraints(parameters, mode, x_90, x_95):
    
    lambdA = parameters[0]
    alpha = parameters[1]
    nu = mode - lambdA * log(alpha)
    
    first = np.abs(cdf_limit(x_90, nu, lambdA, alpha) - 0.90)
    second = np.abs(cdf_limit(x_95, nu, lambdA, alpha) - 0.95)
    
    return [first, second]

def quadratic(parameters):
    return (parameters[0] - 3)**2 

def solve():
    
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
    
def solve_amoroso(mode, x_90, x_95, x_90_initial, x_95_initial):
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
    
if __name__ == '__main__':
    plot_around()
    
#solve()
    
#solve_amoroso(0.0, 0.9, 1.08, 0.6, 0.5)
#plot_amoroso()
    

