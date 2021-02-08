from core import Graph
from distance import euclidean
from scipy.sparse import lil_matrix,vstack
import numpy as np
import math
from matcher import Matcher
#import matcherc

# Identity match
# author: brijneshjain

# ID is a child of matcher
# ID algorithm is used to compute the identity match between two network (i.e. the network itself)
# ID matchining can be used for labelled framework
class ID(Matcher):
    
    def match(self,X,Y,c):
        if(c==True):
            return #provvisrio
            #self.f=matcherc.IDc(X.x,Y.x,True)
        else:
            self.X=X
            self.Y=Y
            nX=X.nodes()
            self.f=range(nX)
