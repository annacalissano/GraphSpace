from core import Graph
from distance import euclidean
from scipy.sparse import lil_matrix,vstack
import numpy as np
import math
from matcher import Matcher

# Identity match
# author: brijneshjain

# ID is a child of matcher
# ID algorithm is used to compute the match between two network (i.e. the network itself)
class ID(Matcher):
    
    def match(self,X,Y):
        self.X=X
        self.Y=Y
        nX=X.nodes()
        self.f=range(nX)