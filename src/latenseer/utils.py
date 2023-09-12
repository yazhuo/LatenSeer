import numpy as np
from scipy import stats

PVAL_SIG = 0.05

def correlation(x, y): 
    """ Compute the covariance between two arrays.
    """
    corr = np.corrcoef(x, y)
    return abs(corr[0][1])


def relative_error(dist1, dist2) -> stats.stats.Ks_2sampResult:
  ks = stats.ks_2samp(dist1, dist2, mode="exact")
  print(f"ks = {ks[0]:.4f}, p-value = {ks[1]:.6f}, equal is {ks[1]>PVAL_SIG}")
  return ks


def removeprefix(string, predix):
    while string.startswith(predix):
        string = string[len(predix):]
    return string


def removesuffix(string, suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string


if __name__ == '__main__':
    test_string = 'compose_post_client'
    print(removeprefix(test_string, '_'))