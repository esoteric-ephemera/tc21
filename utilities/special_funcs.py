import numpy as np

def erf(x):
    """   Abramowitz and Stegun eq. 7.1.26
        maximum absolute error is 1.5 x 10**(-7)
    """
    p = 0.3275911
    a = [0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429]
    """ erf(-x) = - erf(x)  """
    sgnx = np.sign(x)
    t = 1/(1 + p*sgnx*x)
    ttmp = np.ones(t.shape)
    tmp = np.zeros(t.shape)
    for i in range(5):
        ttmp *= t
        tmp += a[i]*ttmp
    return sgnx*(1 - tmp*np.exp(-x**2))
