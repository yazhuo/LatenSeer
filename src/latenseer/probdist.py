import bisect
from collections import Counter
import logging
from random import randint
from typing import List
import matplotlib.pyplot as plt
import numpy as np


class PMF(Counter):
    """A Counter with probabilities"""
    def update_pmf(self, value, weight=None):
        
        if weight is None:
            """Update the PMF with a new value x"""
            self.update([value])
        else:
            self.update({value: weight})
    
    def normalize(self):
        """Normalize the probabilities.
        Remember to normalize before you do any computations!
        """
        total = sum(self.values())
        for key in self:
            self[key] = round(self[key] / total, 6)
        return self
    
    def Cov(self, other):
        """Add two distributions(PMF)"""
        pmf = PMF()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                pmf[key1 + key2] += prob1 * prob2
        return pmf
    
    def __hash__(self):
        """Return an integer hash value"""
        return id(self)
    
    def __eq__(self, other) -> bool:
        return self is other
    
    def ProbLess(self, x):
        """Probability that a random variable is <= x"""
        t = [prob for val, prob in self.items() if val <= x]
        return sum(t)

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))
    
    def plot(self, figname='test.png', ls='-', label=None):
        """Plot the PMF."""
        xs, ys = self.render()
        plt.plot(xs, ys, ls, label=label)
        plt.legend()
        

class CDF(object):
    def __init__(self, obj=None, lp=None):
        
        if obj is not None:
            obj.normalize()
            xs, freqs = zip(*sorted(obj.items()))
            self.latencies = np.asarray(xs)
            self.percentiles = np.cumsum(freqs, dtype=float)
            self.percentiles /= self.percentiles[-1]
        else:
            self.latencies = np.asarray([])
            self.percentiles = np.asarray([])
            if lp is not None:
                self.latencies = np.asarray(lp[0])
                self.percentiles = np.asarray(lp[1])
            else:
                logging.warning('CDF object is empty')
    
    def __str__(self):
        return f'latencies: {self.latencies}\npercentiles: {self.percentiles}'    
        
    def Print(self):
        """Print the values and probs in ascending order."""
        for x, p in zip(self.latencies, self.percentiles):
            print(x, p)
            
    def Value(self, p):
        """Return the InverseCDF(p), the value that corresponds to p."""
        index = np.searchsorted(self.percentiles, p)
        return self.latencies[index]
    
    def Values(self, ps=None):
        """Returns InverseCDF(p), the value that corresponds to probability p.

        If ps is not provided, returns all values.
        """
        if ps is None:
            return self.latencies

        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('Probability p must be in range [0, 1]')
        
        index = np.searchsorted(self.percentiles, ps)
        return self.latencies[index]

    def Percentile(self, p):
        """Return the value that corresponds to percentile p."""
        return self.Value(p / 100.0)
    
    def Percentiles(self, ps):
        """Returns the values that correspond to percentiles ps.
        Args:
            ps: sequence of percentiles in the range [0, 100]
        """
        ps = np.asarray(ps)
        return self.Values(ps / 100)
    
    def Prob(self, x):
        """Returns CDF(x), the probability that corresponds to value x.

        Args:
            x: number

        Returns:
            float probability
        """
        if x < self.latencies[0]:
            return 0
        index = bisect.bisect(self.latencies, x)
        p = self.percentiles[index-1]
        return p
    
    def getPMF(self):
        """Returns the PMF associated with this CDF."""
        pmf = PMF()
        min_latency = self.latencies[0]
        pmf.update({min_latency: self.percentiles[0]})
        
        probability = self.percentiles[0]
        for i in range(1, len(self.latencies)):
            added_probability = self.percentiles[i] - probability
            pmf.update({self.latencies[i]: added_probability})
            probability = self.percentiles[i]
        return pmf
    
    def plot(self, 
             filename='test.png',
             xscale=1,
             ls='-', 
             lw=3,
             color='tab:blue',
             marker='', 
             label=None,
             zorder=1,
             ax=None):
        """Plot the CDF."""
        xs = [l / xscale for l in self.latencies]
        if ax is None:
            plt.plot(xs, self.percentiles, 
                    linestyle=ls, 
                    linewidth=lw,
                    color=color,
                    #  marker=marker,
                    #  markersize=20, 
                    # #  markevery=20,
                    label=label,
                    zorder=zorder)
        else:
            ax.plot(xs, self.percentiles, 
                    linestyle=ls, 
                    linewidth=lw,
                    color=color,
                    #  marker=marker,
                    #  markersize=20, 
                    # #  markevery=20,
                    label=label,
                    zorder=zorder)
        plt.legend()


def ADD_OP(pmf_list: List[PMF]):
    pmf = pmf_list[0]
    for other_pmf in pmf_list[1:]:
        pmf = pmf.Cov(other_pmf)
    return pmf

def FAST_ADD_OP(pmf_list: List[PMF]):
    pass

def MAX_OP(pmf_list: List[PMF], threshold:int = 50):
    cdf_list = [CDF(pmf) for pmf in pmf_list]
    min_x = min([cdf.Percentile(0) for cdf in cdf_list])
    max_x = max([cdf.Percentile(100) for cdf in cdf_list])
    m = len(pmf_list)
    
    n = 100
    step = (max_x - min_x) / n
    values = [min_x + i * step for i in range(n+1)]
    
    probs = []
    for x in values:
        w = np.sum([pmf.ProbLess(x) for pmf in pmf_list]) / m
        probs.append(w)
    
    returned_cdf = CDF(lp=(values, probs))
    returned_pmf = returned_cdf.getPMF()
    
    return returned_pmf

    
def MERGE_OP(pmf_list):
    new_pmf = PMF()
    for pmf in pmf_list:
        new_pmf += pmf
    return PMF(new_pmf)


if __name__ == "__main__":
    l1 = [randint(1, 100) for i in range(1000000)]
    # l2 = [randint(1, 10) for i in range(100)]
    
    p1 = PMF(l1)
    
    cdf1 = CDF(p1)
    # cdf1.Print()
    # print(cdf1.Percentile(10))
    print(cdf1.Percentiles([i for i in range(0, 101)]))
    