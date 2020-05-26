# The following class run the following algorithm approach:
# Take: X1,..,Xn obstervation in our Q quotient space
# Step1: Chose a candidate mean m=X1
# Step2: Align X2,...,Xn wrt m (save the permutation vector)
# Step3: Compute the Frechet Mean
# Step4: m=mean_current_loop
# repreat step2-4 until m is not changing

from core import Graph,GraphSet
from AlignCompute import aligncompute
import numpy as np
import pandas as pd
import random

import copy


class mean_aac(aligncompute):
    
    def __init__(self,graphset,matcher):
        aligncompute.__init__(self,graphset,matcher)
        self.mean=None
        self.var=None
        self.m_dis=None
        self.cov=None
    
    def align_and_est(self, max_iterations=200):
        # Select a Random Candidate:
        first_id=random.randint(0,self.aX.size()-1)
        # first_id = 318
        m_1=self.aX.X[first_id]
        self.f[first_id] = range(self.X.n_nodes)
        # k=200 maximum number of iteration
        for self.k in range(max_iterations):
            print("\n start of iteration: " + str(self.k))
            for i in range(self.X.size()):
                # print('\t already matched: ' + str(i))
                # Align X to Y
                a=self.matcher.align(self.aX.X[i],m_1)
                # Permutation of X to go closer to Y
                self.f[i]=a.f
                #self.aX.X[i]=a.alignedSource()
                #print m_1.x
                #print a.alignedSource().x
                
            m_2=self.est(m_1)
            
            step_range=self.matcher.dis(m_1,m_2)
            
            if(step_range<0.001):
                self.mean=m_2
                # Update aX with the final permutations:
                Aligned=GraphSet()
                # Aligned.add(self.aX.X[0])
                for i in range(self.X.size()):
                    G=self.aX.X[i]
                    G.permute(self.f[i])
                    Aligned.add(G)
                    del G
                self.aX=copy.deepcopy(Aligned)
                del Aligned
                print("Step Range smaller than 0.001")
                return
            else:
                del m_1
                m_1=m_2
                del m_2
                self.f.clear()
        print("Maximum number of iteration reached.")  
        if('m_2' in locals()):
            self.mean=m_2
            # Update aX with the final permutations:
            Aligned = GraphSet()
            Aligned.add(self.aX.X[0])
            for i in range(self.X.size()):
                G = self.aX.X[i]
                G.permute(self.f[i])
                Aligned.add(G)
                del G
            self.aX = copy.deepcopy(Aligned)
            del Aligned
            del m_2,m_1
        else:
            self.mean=m_1
            # Update aX with the final permutations:
            Aligned = GraphSet()
            Aligned.add(self.aX.X[0])
            for i in range(1, self.X.size()):
                G = self.aX.X[i]
                G.permute(self.f[i])
                Aligned.add(G)
                del G
            self.aX = copy.deepcopy(Aligned)
            del Aligned
            del m_1
        
        
            
            
    
    def est(self,m_1):
        m_C=m_1
        for i in range(0,self.X.size()):
            m_C=self.add(1.0/(i+1.0),self.aX.X[i],i/(i+1.0),m_C,self.f[i])
        return m_C
    
    # add function  is the one used for computing the mean
    def add(self,ax,A,ay,B,f):
        # Adjency Matrix: x, y
        y=B.x
        G=copy.deepcopy(A)
        G.permute(f)
        x=G.x

        # coefficients: ax, ay
        # Links
        adjX=G.adj
        adjY=B.adj
        nY=B.n_nodes
        new={}
        fullset=set(x.keys()).union(set(y.keys()))
        
        for i in range(nY):
            if((i,i) in x and (i,i) in y):
                new[i,i]=self.summ(ax,x[i,i],ay,y[i,i])
            elif((i,i) in x and not (i,i) in y):
                new[i,i]=self.summ(ax,x[i,i],ay,None)
            elif((not (i,i) in x) and (i,i) in y):
                new[i,i]=self.summ(ax,None,ay,y[i,i])
                
            #degree=self.X.degree(i)
            linked_nodes=[]
            if(i in adjX and i in adjY):
                linked_nodes=set(adjX[i]).union(set(adjY[i]))
            else:
                if(i in adjX and not i in adjY):
                    linked_nodes=set(adjX[i])
                if(i in adjY and not i in adjX):
                    linked_nodes=set(adjY[i])
                    
            #for j in range(degree):
            for j in linked_nodes:

                if((not (i,j) in y) and (not (i,j) in x)):
                       continue
                elif((i,j) in y and (i,j) in x):
                    new[i,j]=self.summ(ax,x[i,j],ay,y[i,j])
                elif(not (i,j) in y):
                    #new[fi,fj]=self.summ(ax,x[i,j0],ay,[0]*len(x[i,j0]))
                    new[i,j]=self.summ(ax,x[i,j],ay,None)
                    #if(not x.has_key((i,j0))):
                elif(not (i,j) in x):
                    #new[fi,fj]=self.summ(ax,[0]*len(x[i,j0]),ay,y[fi,fj])
                    new[i,j]=self.summ(ax,None,ay,y[i,j])
            newG=Graph(x=new,y=None,adj=None)
        return newG
    
    # Add at y a linear combination of x y=ax*y + ay*x
    def summ(self,ax,x,ay,y): #ax,ay are scalar, x,y are vectors
        if(x is None and y is None):
            return None
        else:
            if(x is None):
                res=[i * ay for i in y]
                return res # ATTENTION: scalar moltiplication of scalar ay with vector y
            else:
                n=len(x)
                if(y is None):
                    y=np.zeros(n)
                res=[]
                for i in range(n):
                    res+=[ax*x[i]+ay*y[i]]
                return res
    

# compute the variance as the distance of all the graphs from the frechet mean
    def variance(self):
        if(self.aX !=None and self.aX.size()!=0):
            if(self.var !=None):
                return self.var
            else:
                if(not isinstance(self.mean,Graph)):
                    self.mean=self.align_and_est()
                n = self.aX.size()
                if(self.m_dis==None):
                    # the variance is computed as a distance between the mean and the sample
                    align_X=GraphSet()
                    for i in range(n):
                        G=copy.deepcopy(self.aX.X[i])
                        G.permute(self.f[i])
                        align_X.add(G)
                        del(G)
                    self.m_dis=self.matcher.dis(align_X,self.mean)
                self.var=0.0
                for i in range(n):
                    self.var+=self.m_dis[i]
                self.var=self.var/n
                return self.var
        else:
                print("Sample of graphs is empty")

# compute the covariance of the aligned dataset to the mean
    def covariance(self):
        if (self.aX != None and self.aX.size() != 0):
            if (self.cov != None):
                return self.cov
            else:
                if (not isinstance(self.mean, Graph)):
                    self.mean = self.align_and_est()
                n = self.aX.size()
                if (self.m_dis == None):
                    # the variance is computed as a distance between the mean and the sample
                    align_X = copy.deepcopy(self.aX)
                    self.cov = np.cov(align_X.to_matrix_with_attr().transpose())
                    # GIANLU: da qui trovare un metodo di storage della covarianza più leggero
        else:
            print("Sample of graphs is empty")


class mean_aac_pred(mean_aac):

    def __init__(self, graphset, matcher, propTraining=0.7):
        # training and calibration sets
        n = graphset.size()
        nt = int(np.floor(n * propTraining))
        idx = np.random.permutation(n)
        idx_train, idx_cal = idx[:nt], idx[nt:]
        mean_aac.__init__(self, graphset.sublist(idx_train), matcher)
        self.X_dev = graphset.sublist(idx_cal)  # is it a copy? no
        self.aX_dev = copy.deepcopy(self.X_dev)
        self.conformal_matrix = None
        # self.conformal_low = None
        # self.conformal_up = None

    # compute conformal prediction regions
    def align_est_and_predRegions(self, alpha=0.1, e=0.00001):
        # check we are using scalar attributes
        self.X.get_node_attr()
        self.X.get_edge_attr()
        n_a = self.X.node_attr
        e_a = self.X.edge_attr
        assert n_a * e_a == 1, "This method assumes that the node or edge attribute is a scalar"

        print('\n COMPUTING THE MEAN:')
        self.align_and_est()
        print('\n MEAN COMPUTED!')

        dataSd = pd.Series(self.aX.to_matrix_with_attr().apply(np.std, axis=0))

        print('\n Aligning calibration set')
        for i in range(self.aX_dev.size()):
            # Align self.aX_dev.X[i] to self.mean
            # match = GAS()
            self.matcher.the_grow_and_set(self.aX_dev.X[i], self.mean)
            self.matcher.match(self.aX_dev.X[i], self.mean)
            self.aX_dev.X[i].permute(self.matcher.f)
            # del match

        print('\n Computing residuals')
        res = abs(self.aX_dev.to_matrix_with_attr() - self.mean.to_vector_with_attributes().iloc[0])
        res_norm = res / (dataSd + e)
        scores = res_norm.max(axis=1)  # L1 norm
        err = np.quantile(scores, 1 - alpha)
        erri = err * dataSd  # err in direction i

        print('\n Computing prediction regions')
        self.conformal_matrix = (self.mean.to_vector_with_attributes() - erri).append(self.mean.to_vector_with_attributes() + erri)
        self.conformal_matrix.index = ["min", "max"]
        self.conformal_matrix = self.conformal_matrix.dropna(axis=1, how='any')  # or [0, 0] instead?

        # TODO Save prediction regions as graphs (and not just as a matrix)
        # self.conformal_low = Graph(x={y: [x] for x in self.conformal_matrix.iloc[0,] for y in self.conformal_matrix.columns},
        #                            y=None, adj=None) # NOT WORKING: columns contains strings :(

