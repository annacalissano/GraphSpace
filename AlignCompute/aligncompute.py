# The following class run the following algorithm approach:
# Take: X1,..,Xn obstervation in our Q quotient space
# Step1: Chose a candidate m=X1
# Step2: Align X2,...,Xn wrt m (save the permutation vector)
# Step3: Compute the statistic tetha you are interested in: Mean, Variance, PCA, etc.
# Step4: m=theta
# repreat step2-4 until m is not changing

from core import Graph
from core import GraphSet
from matcher import Matcher, BK, alignment, GA, ID
from distance import euclidean
import itertools
import copy
import math

class aligncompute(object):
    
    def __init__(self,graphset,matcher):
        if(graphset.size()<2):
            return "Error: I need a GraphSet to compute an estimator! Don't be Scruge, give me at least two observation!"
        self.X=graphset.grow_to_same_size() # Original Dataset
        self.aX=copy.deepcopy(self.X) # Aligned Dataset
        self.f={} # Permutation set of function to define the alignment
        self.matcher=matcher # type of matching desired
    
    # Function aimed at aligning the data wrt the candidate estimator. If the estimator is the mean, wrt the mean, if is
    # the PCA, wrt the geodesic etc.
    def align(self):
        pass
       
    # Function to compute the estimator
    def est(self):
        pass