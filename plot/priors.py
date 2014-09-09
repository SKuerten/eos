import numpy as np

"""
Create function objects representing priors used in eos
"""

class Prior(object):

    def __init__(self, range):

        self.min = range[0]
        self.max = range[1]

    def evaluate(self, x):
        """
        x may be an array.
        """

class Gauss(Prior):

    def __init__(self, range, mu, sigma):
        from scipy.stats.distributions import norm as Gaussian
        Prior.__init__(self, range)
        self.mu = mu
        self.sigma = sigma
        self.norm = 1.0 / np.sqrt(2 * np.pi) / sigma \
                     / (Gaussian.cdf(self.max, self.mu, self.sigma) \
                      - Gaussian.cdf(self.min, self.mu, self.sigma))

    def evaluate(self, x):
        return self.norm * np.exp(- ((x - self.mu) / self.sigma)**2 / 2)

class LogGamma(Prior):

    def __init__(self, range, nu, lamb, alpha):
        from scipy.special import gamma
        Prior.__init__(self, range)
        self.nu = nu
        self.lamb = lamb
        self.alpha = alpha
        self.norm = 1.0 / gamma(alpha) / np.abs(lamb)

        volume = self.__cdf(self.max) - self.__cdf(self.min)
        self.norm /= volume

    def evaluate(self, x):
        std = (x - self.nu) / self.lamb
        return self.norm * np.exp(self.alpha * std - np.exp(std))

    def __cdf(self, x):
        from scipy.special import gammaincc
        if self.lamb > 0:
            return 1 - gammaincc(self.alpha, np.exp((x - self.nu) / self.lamb))
        else:
            return gammaincc(self.alpha, np.exp((x - self.nu) / self.lamb))

class Flat(Prior):

    def __init__(self, range):
        Prior.__init__(self, range)
        self.norm = 1.0 / (self.max - self.min)

    def evaluate(self, x):
        try:
            result = np.zeros(x.shape)
            result += self.norm
            return result
        except:
            return self.norm

class PriorFactory(object):
    """
    Create 1D priors from string like
    a) Parameter: Re{c7}, prior type: flat, range: [-0.75,0.75]
    b) Parameter: CKM::etabar, prior type: Gaussian, range: [0.291,0.471], x = 0.381 +- 0.03
    c) Parameter: mass::c, prior type: LogGamma, range: [1,1.48], x = 1.27 + 0.07 - 0.09, nu: 1.201633227, lambda: 0.1046632085, alpha: 1.921694426");

    """

    def create(self, string_representation):

        s = string_representation
        #  parameter name
        i = s.find('Parameter:')
        j = s.find(',', i)
        par_name = s[i + 11:j]

        # prior type
        i = s.find('type:')
        j = s.find(',', i)
        prior_type = s[i + 6:j]

        # range
        i = s.find('range:')
        j = s.find(',', i)

        min = float(s[i + 8:j])

        i = j
        j = s.find(']',i)
        max = float(s[i + 1:j])

        if prior_type == 'flat':
            return (par_name, Flat((min, max)))

        # extract parameters
        if np.any(prior_type == np.array(('Gaussian', 'LogGamma'))):
            i = s.find('x =')
            j = s.find('+',i)
            x = float(s[i + 3:j])

            #symmetric or not
            if s[j + 1] == '-':
                return (par_name, Gauss((min, max), x, float(s[j + 2:])))
            else:
                i = j
                j = s.find('-', i)
                sigma_up = float(s[i + 1:j])

                i = j
                j = s.find(',', i)
                sigma_down = float(s[i+1:j])

                if prior_type == 'Gaussian':
                    raise KeyError("Asymmetric Gaussian not implemented yet.")

                # now find parameters nu, lambda, alpha
                i = s.find('nu:', j)
                j = s.find(',', i)

                nu = float(s[i + 3:j])

                i = s.find('lambda:', j)
                j = s.find(',', i)
                lamb = float(s[i + 7:j])

                i = s.find('alpha:', j)
                alpha = float(s[i + 6:])

                return (par_name, LogGamma((min, max), nu, lamb, alpha))

        else:
            raise KeyError("Unknown prior type: %s" % prior_type)

        print(par_name, prior_type, min, max)

def test_priors():
    flat = Flat((0, 2))
    print(flat.evaluate(np.array([3.0, 1, 2, 4])))

    gauss = Gauss((-1, 1), 0, 1)
    print((gauss.evaluate(0), Gaussian.cdf(1.5, 1, 0.5) - Gaussian.cdf(0.5, 1, 0.5)))

    f = PriorFactory()
    par_name, prior = f.create("Parameter: Re{c7}, prior type: flat, range: [0,2]")
    assert(prior.evaluate(3.0) == flat.evaluate(3.0))
    assert(par_name == 'Re{c7}')

    par_name, prior = f.create("Parameter: Re{c9}, prior type: Gaussian, range: [-1,1], x = 0.0 +- 1")
    assert(prior.evaluate(0) == gauss.evaluate(0))
    assert(par_name == 'Re{c9}')

    par_name, prior = f.create('Parameter: mass::c, prior type: LogGamma, range: [1,1.48], x = 1.27 + 0.07 - 0.09, nu: 1.201633227, lambda: 0.1046632085, alpha: 1.921694426')

