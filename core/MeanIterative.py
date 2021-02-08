from core import Graph
from core import GraphSet
from matcher import Matcher, alignment, GA# ID , BK,
from distance import euclidean
import itertools
import copy
import math
from sklearn.utils import resample


# MeanIterative Class: bootstrapped version of the Mean Class
# the class take as an input a set of graphs and compute the frechet mean and the variance


# Iterative Mean Algorithm
# Jain, Brijnesh, and Klaus Obermayer. "On the sample mean of graphs." 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008.

class MeanIterative:

    def __init__(self, GraphSet, Matcher):
        self.m_matcher = Matcher # type of alignment
        self.m_sample = GraphSet # graphset
        self.m_C = None # Frèchet Mean
        self.m_dis = None # distance between in graphset and the Frèchet Mean
        self.var = None # Variance

    # compute the mean:
    # Bootstrapped N data from the original dataset
    # select a random candidate
    # align all graph to this and compute a mean
    def mean(self, N=None):
        if (isinstance(self.m_C, Graph)):
            return self.m_C
        else:
            if (self.m_sample != None and self.m_sample.size() != 0):
                n = self.m_sample.size()
                step = True
                if (N == None):
                    f = resample(range(n), replace=True, n_samples=10 * n)
                else:
                    f = resample(range(n), replace=True, n_samples=N)
                # Current loop Mean Candidate
                m_C_old = copy.deepcopy(self.m_sample.X[f[0]])
                for i in range(1, len(f)):
                    # new observation (or resampled)
                    i0 = f[i]
                    # computing the current mean as in Mean object
                    a = self.m_matcher.align(copy.deepcopy(self.m_sample.X[i0]), m_C_old)
                    m_C_curr = a.add(1.0 / (i + 1.0), i / (i + 1.0))
                    # compute the distance from the previous step
                    step_range = self.m_matcher.dis(copy.deepcopy(m_C_old), copy.deepcopy(m_C_curr))
                    if (step_range < 0.01):
                        self.m_C = m_C_curr
                        self.m_C.setFeatures(0)
                        return self.m_C
                        m_C_old = copy.deepcopy(m_C_curr)
                        del (a, m_C_curr)
            self.m_C = copy.deepcopy(m_C_curr)
            self.m_C.setClassLabel(0)
            return self.m_C


# compute the variance as the distance of all the graphs from the frechet mean
def variance(self):
    if (self.m_sample != None and self.m_sample.size() != 0):
        if (self.var != None):
            return self.var
        else:
            if (not isinstance(self.m_C, Graph)):
                self.m_C = self.mean()
                print(self.m_C.x)
            if (self.m_dis == None):
                # the variance is computed as a distance between the mean and the sample
                self.m_dis = self.m_matcher.dis(copy.deepcopy(self.m_sample), copy.deepcopy(self.m_C))
            n = self.m_sample.size()
            self.var = 0.0
            for i in range(n):
                self.var += self.m_dis[i]
            self.var = self.var / n
            return self.var
    else:
        print("Sample of graphs is empty")


# compute the standard deviation
def std(self):
    return math.sqrt(self.variance())


# aligning all the graph to the Frèchet mean and save them in a new set
def align_G(self, *args):
    if (isinstance(args, Graph)):
        if (self.m_C == None):
            return args
        else:
            a = self.m_matcher.align(args, self.m_C)
            return a.alignedSource()
    if (isinstance(args, GraphSet)):
        if (self.m_C == None):
            return args
        else:
            new_a_set = GraphSet()
            i = 0
            while (i == args.size()):
                Gi = args.X[i]
                # add to the new graph set an aligned graph
                new_a_set.add(self.align_G(Gi))
                i += 1
            return new_a_set
