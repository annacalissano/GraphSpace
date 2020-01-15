from core import Graph
from core import GraphSet
import random
import copy
import math

# Mean Class
# the mean class take as an imput a set of graphs and compute the frechet mean and the variance


# Reference:
# Jain, Brijnesh, and Klaus Obermayer. "On the sample mean of graphs." 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008.

class Mean:
    
    # input:
    # -Graphset: a set of graphs as a graphset object
    # -Matcher: the type of alignment to use: either ID or GA
    
    def __init__(self,GraphSet,Matcher):
        self.m_matcher=Matcher # the type of alignment to use
        self.m_sample=GraphSet # the graphset
        self.m_C=None # the Frechet Mean
        self.m_dis=None # the distance of all the graphset wrt to the Frèchet Mean
        self.var=None # the variance
        self.order=None # order of shuffled data index
        
    # Compute the mean:
    # select a random candidate 
    # align all graph to the random cadidate and compute a mean
    def mean(self):
        if(isinstance(self.m_C, Graph)):
            return self.m_C
        else:
            if(self.m_sample !=None and self.m_sample.size()!=0):
                n=self.m_sample.size()
                f=list(range(n))
                random.shuffle(f)
                self.order=f
                # Select as a candidate the first element of the new random permutation of graph
                self.m_C=copy.deepcopy(self.m_sample.X[f[0]])
                # the mean is compute thanks to the alignment function
                for i in range(1,n):
                    i0=f[i]
                    # aign the observation to the candidate mean
                    a=self.m_matcher.align(copy.deepcopy(self.m_sample.X[i0]),self.m_C)
                    # go along the geodesic of a step 1/k
                    self.m_C=a.add(1.0/(i+1.0),i/(i+1.0))
                    del a
                self.m_C.setClassLabel(0)
                return self.m_C
            else: return None

    # Compute the variance
    # the variance is the distance of all the graphs from the Frechet mean 
    def variance(self):
        if(self.m_sample !=None and self.m_sample.size()!=0):
            if(self.var !=None):
                return self.var
            else:
                if(not isinstance(self.m_C, Graph)):
                    self.m_C=self.mean()
                if(self.m_dis==None):
                    # the variance is computed as a distance between the mean and the sample
                    self.m_dis=self.m_matcher.dis(copy.deepcopy(self.m_sample),self.m_C)
                n=self.m_sample.size()
                self.var=0.0
                for i in range(n):
                    self.var+=self.m_dis[i]
                self.var=self.var/n
                return self.var
        else: 
                print("Sample of graphs is empty")
      
    # compute the standard deviation
    def std(self):
            return math.sqrt(self.variance())
    
    # aligning all the graph to the Frechet mean and save them in a new set
    def align_G(self,*args):
            if(isinstance(args,Graph)):
                if(self.m_C==None):
                    return args
                else:
                    a=self.m_matcher.align(args,self.m_C)
                    return a.alignedSource()
            if(isinstance(args,GraphSet)):
                if(self.m_C==None):
                    return args
                else:
                        new_a_set=GraphSet()
                        i=0
                        while(i==args.size()):
                            Gi=args.X[i]
                            # add to the new graph set an aligned graph
                            new_a_set.add(self.align_G(Gi))
                            i+=1
                        return new_a_set
