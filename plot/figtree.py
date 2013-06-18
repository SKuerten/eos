
import numpy as np
import ctypes as C

#automatically loads ANN library if needed
#both have to be available on $LD_LIBRARY_PATH
_lib = C.cdll.LoadLibrary('libfigtree.so')

#returns an integer
_lib.figtree.restype = C.c_int

#many arguments, all of basic type
#important to ensure contiguous C-style memory
_lib.figtree.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, 
                         np.ctypeslib.ndpointer(dtype='float64', flags="CONTIGUOUS,ALIGNED"), C.c_double,
                         np.ctypeslib.ndpointer(dtype='float64', flags="CONTIGUOUS,ALIGNED"), 
                         np.ctypeslib.ndpointer(dtype='float64', flags="CONTIGUOUS,ALIGNED"), C.c_double,
                         np.ctypeslib.ndpointer(dtype='float64', flags="CONTIGUOUS,ALIGNED"), 
                         C.c_int, C.c_int, C.c_int, C.c_int ]

class FigtreeConfig(object):
    """
    Constants for using the truncation, parameter selection and evaluation methods
    """
    def __init__(self):
        self.truncation = {"max":0, "point":1, "cluster":2}
        self.parameter = {"uniform":0, "non-uniform":1}
        self.evaluation = {"direct":0, "IFGT":1, "direct-tree":2, "IFGT-tree":3, "auto":4}

      

def figtree(samples, targets, weights, bandwidth=0.8, epsilon=1e-2,
            eval="auto", trunc="cluster", param="non-uniform", verbose=False):
    """
    Python wrapper of the figtree method by Vlad Morariu.
    
    
    samples: one per row. Number of columns is the dimension of sample space.
    target: one position per row
    weights: dimensional array of weights, 
    or one row per set of weights, 
    i.e. for three distinct weights for same point, need three rows. 
    
    returns: kernel density estimate at target points
    """
    
    #check input
    try:
        if not samples.flags['C_CONTIGUOUS']:
            raise Exception("figtree: samples are not contiguous, use numpy.ascontiguousarray!")
        if not targets.flags['C_CONTIGUOUS']:
            raise Exception("figtree: targets are not contiguous, use numpy.ascontiguousarray!")
        if not weights.flags['C_CONTIGUOUS']:
            raise Exception("figtree: weights are not contiguous, use numpy.ascontiguousarray!")
    except AttributeError:
        raise Exception("One of the input arrays is not a numpy array!")            

    
    #use same notation as in figtree.h
    
    one_dim = len(samples.shape)==1
    if one_dim:
        d=1
    else:
        d = samples.shape[1]
        
    #number of samples    
    N = samples.shape[0]
    
    #number of target points
    M = targets.shape[0]
    
    
    
    if not one_dim:
        #dimensions of samples and target have to match
        assert(targets.shape[-1] == d)
    else:
        assert(len(targets.shape)== 1)
    #one weight for each source sample
    
    
    
    #number of weights per points 
    if len(weights.shape)==1:
        W = 1
        assert(weights.shape[0] == N)
    else:
        #assume one row per weights if several exist
        W = weights.shape[0]
        assert(weights.shape[-1] == N)
    
    
    
    conf = FigtreeConfig()
    
    #define array to hold densities at target points
    g = np.zeros((W*M,))
    assert(g.flags['C_CONTIGUOUS'])
    
    if verbose:
        print("Dimension = %d, N samples= %d, M targets= %d, W weights= %d" % (d,N, M, W))
        print("bandwidth h=%g , weights= %d, epsilon=%g "%(bandwidth, W, epsilon))
        print("Samples: %s, targets: %s, weights: %s" % (samples.shape, targets.shape, weights.shape))
    
    #perform calculation
    err = _lib.figtree(d, N, M, W, samples, bandwidth, weights, targets, epsilon, 
                 g, conf.evaluation[eval], conf.parameter[param], conf.truncation[trunc], int(verbose))

    if err < 0:
        raise Exception("figtree failed!")
    
    return g
    
    
def test_figtree():
      """
      Redo example from sample.cpp, distributed with the figtree source,
      and check that the same numbers come out using the python interface
      """
    
      x = np.array([0.7165, 0.5113, 0.7764, 0.4893, 0.1859, 0.7006, 0.9827,
                0.8066, 0.7036, 0.4850, 0.1146, 0.6649, 0.3654, 0.1400,
                0.5668, 0.8230, 0.6739, 0.9994, 0.9616, 0.0589, 0.3603,
                0.5485, 0.2618, 0.5973, 0.0493, 0.5711, 0.7009, 0.9623,
                0.7505, 0.7400, 0.4319, 0.6343, 0.8030, 0.0839, 0.9455,
                0.9159, 0.6020, 0.2536, 0.8735, 0.5134, 0.7327, 0.4222,
                0.1959, 0.1059, 0.3923, 0.1003, 0.6930, 0.2069, 0.6094,
                0.1059, 0.0396, 0.2093, 0.9693, 0.1059, 0.3029, 0.3069,
                0.9692, 0.6029, 0.2222, 0.2059, 0.3059, 0.6092, 0.2133,
                0.9614, 0.0721, 0.5534, 0.2920, 0.8580, 0.3358, 0.6802,
                0.2473, 0.3527, 0.1879, 0.4906, 0.4093, 0.4635, 0.6109,
                0.1865, 0.0395, 0.5921, 0.1853, 0.9963, 0.1953, 0.7659,
                0.0534, 0.3567, 0.4983, 0.4344, 0.5625, 0.6166, 0.1133,
                0.8983, 0.7546, 0.7911, 0.8150, 0.6700, 0.2009, 0.2731,
                0.6262, 0.5369, 0.0595, 0.0890, 0.2713, 0.4091, 0.4740,
                0.1332, 0.6926, 0.0009, 0.1532, 0.9632, 0.3521, 0.9692,
                0.9623, 0.3532, 0.7432, 0.0693, 0.2336, 0.6022, 0.2936,
                0.3921, 0.6023, 0.6323, 0.9353, 0.3963, 0.2835, 0.9868,
                0.2362, 0.6682, 0.2026, 0.0263, 0.1632, 0.9164, 0.1153,
                0.9090, 0.5962, 0.3290, 0.4782, 0.5972, 0.1614, 0.8295])
      x.resize((20, 7))
      
      y = np.array([0.9561, 0.5955, 0.0287, 0.8121, 0.6101, 0.7015, 0.0922,
                0.4249, 0.3756, 0.1662, 0.8332, 0.8386, 0.4516, 0.9566,
                0.1472, 0.8699, 0.7694, 0.4442, 0.6206, 0.9517, 0.6400,
                0.0712, 0.3143, 0.6084, 0.1750, 0.6210, 0.2460, 0.5874,
                0.5061, 0.4648, 0.5414, 0.9423, 0.3418, 0.4018, 0.3077,
                0.4116, 0.2859, 0.3941, 0.5030, 0.7220, 0.3062, 0.1122,
                0.4433, 0.4668, 0.0147, 0.6641, 0.7241, 0.2816, 0.2618,
                0.7085, 0.7839, 0.9862, 0.4733, 0.9028, 0.4511, 0.8045,
                0.8289, 0.1663, 0.3939, 0.5208, 0.7181, 0.5692, 0.4608,
                0.4453, 0.0877, 0.4435, 0.3663, 0.3025, 0.8518, 0.7595])
      y.resize((10, 7))
      
      q = np.array([0.2280, 0.4496, 0.1722, 0.9688, 0.3557, 0.0490, 0.7553,
                     0.8948, 0.2861, 0.2512, 0.9327, 0.3353, 0.2530, 0.2532,
                     0.3352, 0.7235, 0.2506, 0.0235, 0.1062, 0.1061]) #, 0.7234, 0.1532])
      
      #bandwidth
      h = 0.8
      
      epsilon = 1e-2
      
      import time
      start_time = time.time()
      target_densities = figtree(x, y, q, h, epsilon)
      end_time = time.time()
      
      print("Used %f time" % (end_time-start_time) )
      
      #results from C compilation and shell output
      direct_results = np.array([
       1.0029,
       1.8818,
       1.1551,
       2.7236,
       1.9211,
       2.3743,
       2.1198,
       1.3643,
       2.3783,
       2.3393,
       ])
      print(abs(target_densities - direct_results))
      if (abs(target_densities - direct_results) > 5e-5).any():
          print("TEST FAILED!")
      else:
          print("TEST PASSED!")
          
def test_figtree_1D():
    """
    Make sure that it works also in one dimension
    """
    x = np.array([1, 1.3, 1.4, 1.2, 1.8, 3.2])
    y = np.zeros((2,))
    y[0]=1
    y[1]=5
    #causes rubbish to be passed to C. Datatype is int!
#    y = np.array([1, 5])
    #however this doesn't. double 
    y  = np.array([1, 5],dtype='float64')
    q = np.ones(x.shape)
    q /= sum(q)
    h = 0.9
    
#    print("contiguous: %d"% x.flags['C_CONTIGUOUS'])
#    figtreeChooseParametersNonUniform() chose p=1, k=6.
#Eval IFGT(h= 9.00e-01, pMax= 1, K= 6, r= 1.93e+00, rx= 0.00e+00, epsilon= 1.00e-02, bound = 0.00e+00)
    target_densities = figtree(x, y, q, h,epsilon=1e-2, eval="auto", verbose=True)
    
    print(target_densities)
    
    c_results = np.array([0.6868666, 0.0030526])
    try:
        if (abs(target_densities - c_results) > 5e-5).any():
            print("1D TEST (a) FAILED!")
        else:
            print("1D TEST (a) PASSED!")
    except ValueError:
        pass        
    #now with more target points
    y2 = np.array([1, 1.3, 2.1, 2.8, 3.6, 4])
    target_densities = figtree(x, y2, q, h,epsilon=1e-2, eval="auto", verbose=False)
    print(target_densities)
    c_results = np.array([0.6868666,     0.7693905 ,    0.4519399 ,    0.2205926 ,    0.1398451 ,    0.0756315])
    if (abs(target_densities - c_results) > 1e-4).any():
        print("1D TEST (b) FAILED!")
    else:
        print("1D TEST (b) PASSED!")
    
    
def test_figtree_2D():
    """
    Make sure that it works also in two dimensions
    """
    x = np.array([1, 1.3,
                  1.4, 1.2,
                  1.8, 3.2,
                  2.2, 1.3])
    x.resize((4,2))
    y = np.array([1, 1,
                  1.5, 1.5,
                  2, 3])
    y.resize((3, 2))
    q = np.ones(x.shape[0])
    q /= sum(q)
    h = 1.1
    
    target_densities = figtree(x, y, q, h,epsilon=1e-3)
    print(target_densities)
    
    c_results = np.array([0.5172876,     0.6095191,     0.2790023])
    if (abs(target_densities - c_results) > 5e-7).any():
        print("2D TEST FAILED!")
    else:
        print("2D TEST PASSED!")
    
def test_distortion():
    """
    Verify that the density is completely  distorted when the parameter ranges are different.
    To use figtree, transform coordinates to the unit hypercube first 
    """
    from numpy import random
    import pylab as P
    
    nSamples = 1000
    d=2
    
    nTargetPerAxis = 30
    
    #create fake data
    x1 = np.random.normal(5, 2, (nSamples))
    x2 = np.random.normal(0.33, 0.01, (nSamples))
    x = np.c_[x1.ravel(),x2.ravel()]
    q = np.ones(x.shape[0])
    h = 1
    
    x1_min = min(x1); x1_max = max(x1)
    x2_min = min(x2); x2_max = max(x2)
    
    #define grid
    grid1, grid2 = np.meshgrid(np.linspace(x1_min, x1_max, nTargetPerAxis),np.linspace(x2_min,x2_max , nTargetPerAxis) )
    y = np.c_[grid1.ravel(), grid2.ravel()]
    
    print(y.shape)
    
    target_densities = figtree(np.ascontiguousarray(x), np.ascontiguousarray(y), q, h,epsilon=1e-3)
    
    Z = np.reshape(target_densities.T, grid1.shape)
            
    Z = np.flipud(np.fliplr(np.rot90(Z,k=3)))
    P.imshow(     Z,
        extent=[x1_min, x1_max, x2_min, x2_max],
        interpolation='nearest'
        
        )
    P.axis('tight')
    P.savefig("original_coord.png")
    P.clf()

    #good ol' histogram
    H, xedges, yedges = np.histogram2d(x1, x2 , bins=10)
    H = np.flipud(np.fliplr(np.rot90(H,k=3)))
    P.imshow(     H,
             extent = (xedges[0], xedges[-1], yedges[0], yedges[-1]), 
             interpolation='nearest'
    )
    P.axis('tight')
    P.savefig("hist.png")
    P.clf()
    
    #now apply linear trafo to unit hypercube
    x1  = (x1 - x1_min)/ (x1_max - x1_min)
    x2  = (x2 - x2_min)/ (x2_max - x2_min)
    x = np.c_[x1.ravel(),x2.ravel()]
    
    h =0.1


    grid1, grid2 = np.meshgrid(np.linspace(0, 1, nTargetPerAxis),np.linspace(0,1 , nTargetPerAxis) )
    y = np.c_[grid1.ravel(), grid2.ravel()]
    target_densities = figtree(np.ascontiguousarray(x), np.ascontiguousarray(y), q, h,epsilon=1e-3)

    Z = np.reshape(target_densities.T, grid1.shape)
            
    Z = np.flipud(np.fliplr(np.rot90(Z,k=3)))
    P.imshow(     Z,
        interpolation='nearest',
        extent=[x1_min, x1_max, x2_min, x2_max]
        )
    P.axis('tight')
    P.savefig("unit_cube.png")
    
def  test_exceptions():
    """
    Create bad calls to figtree and make sure they are caught
    """
    x = np.array([1, 1.3, 1.4, 1.2, 1.8, 3.2])
    y = np.zeros((2,))
    y[0]=1
    y[1]=5
   
    #however this doesn't. double 
    y  = np.array([1, 5],dtype='float')
    q = np.ones(x.shape)
    q /= sum(q)
    h = 0.9
    
    try:
         #causes rubbish to be passed to C. Datatype is int!
        y2 = np.array([1, 5], dtype='int')
        target_densities = figtree(x, y2, q, h,epsilon=1e-2, eval="auto", verbose=True)
        assert(False)
    except C.ArgumentError:
        pass
    
    try:
        #wrong shape/dimensions
         y3  = np.array([1, 5, 123, 112,12,11],dtype='float')
         y3.resize((3,2))
         target_densities = figtree(x, y3, q, h,epsilon=1e-2, eval="auto", verbose=True)
         assert(False)
    except AssertionError:
        pass
    
    try:
        #weights off
        x2 = x[0:2]
        target_densities = figtree(x2, y, q, h,epsilon=1e-2, eval="auto", verbose=True)
        
    except AssertionError:
        pass
    
    #not contiguous

def duplicate_weights(array_in):
    """
    Filter out any duplicate d-dimensional samples, assuming they are always next to each other
    counting in rows
    Format: 
    x1 x2 x3 ... xd
    x1 x2 x3 ... xd 
    
    Returns two arrays (i, n_i): 
    the indices of the __last__ occurence of a unique sample,
    and its multiplicity 
    """
    
    #add column for multiplicity
    index_array = np.empty((array_in.shape[0],),dtype=np.int)
    multiplicities = np.empty((array_in.shape[0],),dtype=np.int)
    
    #count unique entries
    index = 0
    
    #keep track of how often same event is seen
    counter = np.zeros((1,)) 
    
    for  row in range(array_in.shape[0]-1):
        counter += 1
        
        #unique entry
        if (array_in[ row+1] != array_in[ row]).any():
            index_array[index] = row
            multiplicities[index] = counter
            index += 1
            counter = 0
        
    #add last element
    counter += 1
    index_array[index] = row + 1
    multiplicities[index] = counter
    index += 1
    
    #crop result array
    index_array.resize((index,) )
    multiplicities.resize((index,))
    
    return (index_array, multiplicities)    

def test_duplicate_weights():
    
    data = np.array( [[0,1], [0,1], [0,1], [0.3, 0.8], [0.3, 0.8], [3,1], [3,2], [3,2]] )
    
    index_array, q = duplicate_weights(data) 
    assert( (index_array == np.array([ 2.,  4.,  5.,  7.])).all())
    assert( (q == np.array([3, 2, 1, 2])).all() )
    
    #only remove last element
    
    data = np.array( [[0,1], [0,1], [0,1], [0.3, 0.8], [0.3, 0.8], [3,1], [3,2]] )
    
    index_array, q = duplicate_weights(data) 
    assert( (index_array == np.array([2.,  4.,  5.,  6.])).all())
    assert( (q == np.array([3, 2, 1, 1])).all())
    
    
def correct_weights(samples, bandwidth, ranges = None, filter=True):
    """
    Update weights such that each sample has 
    a weight of one on an allowed parameter range.
    Filter duplicates if required.
    One sample per row
    ranges: dx2 array, with (min, max) in each row
    """
    
    from scipy.stats import norm
    
    if filter:
        index_array, q = duplicate_weights(samples)
    else:
        index_array = np.arange(samples.shape[0])
        q = np.ones((samples.shape[0],))
        
    #dimensionality of sample vector
    one_dimensional = False
    try:
        d = samples.shape[1]
    except IndexError:
        d = 1
        one_dimensional = True
    
        
    #variance of Gaussian
    sigma = bandwidth / np.sqrt(2.0)

    if ranges is not None:
        for index in index_array:
            #find normalization constant
            c = 1.0
            if one_dimensional:
                c *= norm.cdf(ranges[1], loc=samples[index], scale=sigma) - \
                      norm.cdf(ranges[0], loc=samples[index], scale=sigma)
            else:             
                for i in range(d):
                    c *= norm.cdf(ranges[i,1], loc=samples[index, i], scale=sigma) - \
                          norm.cdf(ranges[i,0], loc=samples[index, i], scale=sigma)
            q[index] /= c
        
    #now apply normalization of Gaussian (1D) and N samples
    q /= (bandwidth * np.sqrt(np.pi))**d * len(q)
    return q

def test_correct_weights():
    
    #draw data from an exponential distribution
    from scipy.stats import expon
    import pylab as P

    scale = 1.0
    
    data = expon.rvs(scale=scale, size=8000)
    
    bandwidth = 20/np.sqrt(data.shape[0])
    
    range = np.array([0, 7])
    
    
    
    n = 500
    y = np.linspace(range[0], range[1], num=n)
    
    eps = 1e-5

    ## Use corrected samples    
    q = correct_weights(data, bandwidth, range, filter=False)
#    q /= np.sum(q) 
#    for sample, weight in zip(data, q):
#        print("(%g,  %g)" % (sample, weight))
    
    target_densities1 = figtree(data, y, q, bandwidth, epsilon=eps, eval="auto", verbose=True)

#    for pos, density in zip(y , target_densities):
#        print("(%g,  %g)" % (pos, density))

    #now try again with uncorrected densities
    q = np.ones(data.shape)
#    q /=  np.sum(q)
    target_densities2 = figtree(data, y, q, bandwidth, epsilon=eps, eval="auto", verbose=True)


    #true density: fix normalization far from zero
#    const1 = target_densities1[50*n/100]/expon.pdf(y[50*n/100], scale=scale)
#    const2 = target_densities2[50*n/100]/expon.pdf(y[50*n/100], scale=scale)
    
    #sample mean for the integral
#    const1 = np.mean(target_densities1) * (range[1] - range[0] ) #(y[-1] - y[0])
#    const2 = np.mean(target_densities2) * (range[1] - range[0] ) #(y[-1] - y[0])

#    print("constants: " + str((const1, const2)))
    print("Smallest sample at %g"%min(data)) 
    
    #plot the exponential density with max. likelihood estimate of the scale
    P.plot(y, expon.pdf(y, scale=np.mean(data)))        
    P.plot(y, target_densities1 , 'ro')
#    P.plot(y, target_densities2, 'k--')
#    P.legend(("true", "corrected", "uncorrected"), shadow = True)
    P.title("Gaussian Kernel Density Estimation")
#    P.xlim(0,3*bandwidth)
    P.show()
    P.savefig("KDE_50000_h-0.05.eps")
  
    
if __name__ == '__main__': 
#    test_figtree_1D()
#    test_figtree_2D()   
#    test_figtree()
#    test_distortion()
#    test_exceptions()
    
#    test_duplicate_weights()
    test_correct_weights()