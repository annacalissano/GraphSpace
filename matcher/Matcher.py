from core import Graph, GraphSet
from distance import euclidean
from scipy.sparse import lil_matrix, vstack
from matcher import alignment
import numpy as np
import copy
import math

# Matcher: father class of all the method used for matching two nodes
# The method match is implemented in child class such as BK,GA,ID

class Matcher(object):
    
    def __init__(self,measure=None,X=None,Y=None,f=None):
        if(measure==None):
            self.measure=euclidean()
        else:
            self.measure=measure
        # if not weighted -> weights are an n*n adjacency matrix of 1 elements
        # if is weighted -> weights are an  n*n adjacency matrix as received in input
        self.X=None
        self.Y=None
        # f represent the permutation of X to get to Y
        self.f=None

    # Core method of the Matcher: initialize the best matching permutation of nodes between X and Y
    def match(self,measure,X,Y):
        pass
    
    # Grows both network to have the same size
    def the_grow_and_set(self,X,Y):
        nX=X.nodes()
        nY=Y.nodes()
        n=max(nX,nY)
        aX=copy.deepcopy(X)
        aY=copy.deepcopy(Y)
        if(nX<n):
            aX.grow(n)
            self.X=aX
        else: self.X=aX
        if(nY<n):
            aY.grow(n)
            self.Y=aY
        else: self.Y=aY    
        
    
    
    # Clone the matcher
    def clone(self):
        M=copy.deepcopy(self)
        M.measure=self.measure
        M.X=None
        M.Y=None
        M.f=None
        return M
    
    # Aligning function: is aligning two network
    # See Alignment class for details
    def align(self,X,Y):
        self.the_grow_and_set(X,Y)
        self.match(self.X,self.Y)
        a=alignment(self.X,self.Y,self.f,self.measure)
        return a


    
    # Computing similarity between two graph
    # the_sim is the father function calling node_sim and edge_sim
    # GraphSpace framework allows for different type of attributes on nodes and edges,
    # so two different sim are implemented
    # see measure.node_sim and measure.edge_sim in measure for details
    def the_sim(self,X,Y):
        self.the_grow_and_set(X,Y)
        aX=copy.deepcopy(self.X)
        aY=copy.deepcopy(self.Y)
        nX=aX.nodes()
        nY=aY.nodes()
        x=aX.matrix()
        y=aY.matrix()
        adjX=aX.adjList()
        sim=0
        for i in range(nX):
            fi=self.f[i]
            if(fi<nY):
                sim+=self.measure.node_sim(x[i,i],y[fi,fi])#w[i,i]
                degX=aX.degree(i)
                for j in range(degX):
                    j0=adjX[i][j]
                    if(self.f[j0]<nY):
                        sim+=self.measure.edge_sim(x[i,j0],y[fi,self.f[j0]]) #w[i,j0]
        return sim
    
    # Computing distance between two graph
    # the_dis is the father function calling node_dis and edge_dis
    # GraphSpace framework allows for different type of attributes on nodes and edges,
    # so two different dis are implemented
    # see node_dis and edge_dis in measure for details
    def the_dis(self,X,Y):
        # match gives back the best combination of nodes
        self.the_grow_and_set(X,Y)
        aX=copy.deepcopy(self.X)
        aY=copy.deepcopy(self.Y)
        self.match(aX,aY)
        n=aX.n_nodes
        x=aX.matrix()
        y=aY.matrix()
        dis=0
        for i in range(n):
            fi=self.f[i]
            dis+=self.measure.node_dis(x[i,i],y[fi,fi])
            for j in range(i+1,n):
                fj=self.f[j]
                if((i,j) in x and (fi,fj) in y):
                    dis+=self.measure.edge_dis(x[i,j],y[fi,fj])
                else:
                    if((i,j) in x):
                        dis+=self.measure.edge_dis(x[i,j],0)
                    elif((fi,fj) in y):
                        dis+=self.measure.edge_dis(y[fi,fj],0)

                if((j,i) in x and (fj,fi) in y):
                    dis+=self.measure.edge_dis(x[j,i],y[fj,fi])
                else:
                    if((j,i) in x):
                        dis+=self.measure.edge_dis(x[j,i],0)
                    elif((fj,fi) in y):
                        dis+=self.measure.edge_dis(y[fj,fi],0)
        return dis

            
    # sim: take as an imput graph, graph sets or both and compute the sim using the_sim function        
    def sim(self,*args):
        l=len(args)
        if(l>2):
            print('Hi, consider different input, such as two graphs')
            return None
        # or graphset or single graph
        if(l==1):
            if(isinstance(args[0],GraphSet)):
                ########
                # set of graph
                n=len(args[0].X)
                s=lil_matrix((n,n))
                for i in range(n):
                    Xi=args[0].X[i]
                    s[i,i]=self.sim(Xi)
                    for j in range(i+1,n):
                        s[i,j]=self.sim(Xi,args[0].X[j])
                        s[j,i]=s[i,j]
                return s
                ########
            else:
                ########
                # single graph
                nZ=args[0].nodes()
                x=args[0].matrix()
                adjZ=args[0].adj
                length=0
                for i in range(nZ):
                    length+=self.measure.node_sim(x[i,i],x[i,i])
                    deg=args[0].degree(i)
                    for j in range(deg):
                        j0=adjZ[i][j]
                        length+=self.measure.edge_sim(x[i,j0],x[i,j0])
                return length
                ########
        else:
            if(isinstance(args[0], Graph) and isinstance(args[1], Graph)):
                ########
                # two graphs function
                #self.match(args[0],args[1])
                s=self.the_sim(args[0],args[1])
                return s
                ########
            else:
                ########
                # graph and a set of graphs
                if(isinstance(args[0], Graph) and isinstance(args[1], GraphSet)):
                    n=len(args[1].X)
                    s=np.zeros(n)
                    for i in range(n):
                        s[i]=self.sim(args[0],args[1].X[i])
                    return s
                if(isinstance(args[1], Graph) and isinstance(args[0], GraphSet)):
                    n=len(args[0].X)
                    s=[]
                    for i in range(n):
                        s[i]=self.sim(args[1],args[0].X[i])
                    return s
                ########
                
    # dis: take as an imput graph, graph sets or both and compute the dis using the_dis function               
    def dis(self,*args):
        l=len(args)
        if(l>2):
            print('Hi, consider different input, such as two graphs')
            return None
        # or graphset or single graph
        if(l==1):
            if(isinstance(args[0],GraphSet)):
                ########
                # set of graph
                n=len(args[0].X)
                d=lil_matrix((n,n))
                for i in range(n):
                    Xi=args[0].X[i]
                    d[i,i]=self.dis(Xi)   # shouldn't we put it directly to 0 by definition?
                    for j in range(i,n):
                        d[i,j]=self.dis(Xi,args[0].X[j])
                        d[j,i]=d[i,j]
                return d
                ########
            else:
                ########
                # single graph
                print('I am sorry, No distance for single graph!')
                return 0
                ########
        else:
            if(isinstance(args[0], Graph) and isinstance(args[1], Graph)):
                ########
                # two graphs function
                #self.match(args[0],args[1])
                #self.X=args[0]
                #self.Y=args[1]
                d=self.the_dis(args[0],args[1])
                return d
                ########
            else:
                ########
                # graph and a set of graphs
                if(isinstance(args[0], Graph) and isinstance(args[1], GraphSet)):
                    n=len(args[1].X)
                    d=np.zeros(n)
                    for i in range(n):
                        d[i]=self.dis(args[0],args[1].X[i])
                    return d
                if(isinstance(args[1], Graph) and isinstance(args[0], GraphSet)):
                    n=len(args[0].X)
                    d=np.zeros(n)
                    for i in range(n):
                        d[i]=self.dis(args[1],args[0].X[i])
                    return d
                ########            
                
