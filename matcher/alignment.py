from core import Graph
from distance import euclidean
import numpy as np
import copy

# Given the optimal permutation of nodes computed with the selected matcher
# the alignment class is managing two graphs and their alignment according to the permutation

class alignment:
    
    # We Align X to Y with f permutation of Y
    # input:
    # -X: graph to align
    # -Y: graph (this one stay fix)
    # -f: permutation of nodes (e.g. [2,0,1] for a three nodes network)
    # - measure: the type of distance to use
    
    def __init__(self,X,Y,f,measure):
        self.X=X # original graph to be aligned
        self.Y=Y #target original graph
        self.f=f # permuting vector
        self.measure=measure # the distance between networks
        self.aY=copy.deepcopy(Y) #to aligned y
        #self.aY=self.aY.grow(self.X.nodes()) # aligned y
        self.aX=None # to aligned y
        self.alignedSource()


    
    
    # add function:
    # performing a linear combination of two vectors with given weights
    # input:
    # - ax: scalar coefficient of x
    # - x: vector [list type]
    # - ay: scalar coefficient of y
    # - y: vector [list type]
    # Output:
    # - res: scalar representing the result of the addition
    def summ(self,ax,x,ay,y):
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
    
    # Aligning function:
    # Aligning X to Y using f permutation
    def alignedSource(self):
        if(self.aX is None):
            self.aX=copy.deepcopy(self.X)
            self.aX.grow(self.Y.nodes())
            self.aX.permute(self.f)
        return self.aX
    
    # Define or return the alignment source
    def alignedTarget(self):
        return self.aY
    
    # Compute the distance between the two aligned graphs
    # Output:
    # - dis: scalar representing the distance
    # def dis(self):
    #     self.alignedSource()
    #     nX=self.aX.nodes()
    #     print("Number of nodes"+str(nX))
    #     dis=0.0
    #     # the two adjency matrix
    #     x=self.aX.matrix()
    #     y=self.aY.matrix()
    #     adjX=self.aX.adj
    #     adjY=self.aY.adj
    #     for i in range(nX):
    #         if((i,i) in x and (i,i) in y):
    #             dis+=self.measure.node_dis(x[i,i],y[i,i])
    #         elif((i,i) in x and not (i,i) in y):
    #             dis+=self.measure.node_dis(x[i,i],[0])
    #         elif((not (i,i) in x) and (i,i) in y):
    #             dis+=self.measure.node_dis([0],y[i,i])
    #         linked_nodes=[]
    #         if(i in adjX and i in adjY):
    #             linked_nodes=set(adjX[i]).union(set(adjY[i]))
    #         else:
    #             if(i in adjX and not i in adjY):
    #                 linked_nodes=set(adjX[i])
    #             if(i in adjY and not i in adjX):
    #                 linked_nodes=set(adjY[i])
    #         for j in linked_nodes:
    #             # Both edges don't exist in both networks (impossible)
    #             if((not (i,j) in y) and (not (i,j) in x)):
    #                    continue
    #             # Both edges exist in both networks
    #             elif((i,j) in y and (i,j) in x):
    #                 dis+=self.measure.edge_dis(x[i,j],y[i,j])
    #             elif(not (i,j) in y):
    #                 dis+=self.measure.edge_dis(x[i,j],[0])
    #             elif(not (i,j) in x):
    #                 dis+=self.measure.edge_dis([0],y[i,j])
    #     return dis

    # Computing distance between two graph
    # the_sim is the father function calling node_dis and edge_dis
    # GraphSpace framework allows for different type of attributes on nodes and edges,
    # so two different sim are implemented
    # see node_dis and edge_dis in measure for details
    def dis(self):
        aX = copy.deepcopy(self.X)
        aY = copy.deepcopy(self.Y)
        n = aX.n_nodes
        x = aX.matrix()
        y = aY.matrix()
        dis = 0
        for i in range(n):
            fi = self.f[i]
            dis += self.measure.node_dis(x[i, i], y[fi, fi])
            for j in range(i + 1, n):
                fj = self.f[j]
                if ((i, j) in x and (fi, fj) in y):
                    dis += self.measure.edge_dis(x[i, j], y[fi, fj])
                else:
                    if ((i, j) in x):
                        dis += self.measure.edge_dis(x[i, j], 0)
                    elif ((fi, fj) in y):
                        dis += self.measure.edge_dis(y[fi, fj], 0)

                if ((j, i) in x and (fj, fi) in y):
                    dis += self.measure.edge_dis(x[j, i], y[fj, fi])
                else:
                    if ((j, i) in x):
                        dis += self.measure.edge_dis(x[j, i], 0)
                    elif ((fj, fi) in y):
                        dis += self.measure.edge_dis(y[fj, fi], 0)
        return dis

    # Compute the similarity between the two aligned graphs
    # Output:
    # - sim: scalar representing the similarity
    def sim(self):
        nX=self.X.nodes()
        x=self.X.matrix()
        y=self.aY.matrix()
        adjX=self.X.adjList()
        sim=0
        for i in range(nX):
            fi=self.f[i]
            sim+=self.measure.node_sim(x[i,i],y[fi,fi])
            degX=self.X.degree(i)
            for j in range(degX):
                j0=adjX[i][j]
                fj=self.f[j0]
                sim+=self.measure.edge_sim(x[i,j0],y[fi,fj])
        return sim
       
    
    
    # add function: this is a very important function used to compute the geodesic between
    # the two aligned graphs and proceeding ax and ay along the geodesic
    # it is an addition extended to the concept of aligned graphs
    # input:
    # - ax: scalar weight of network X
    # - ay: scalar weight of network Y
    # Output:
    # - newG: the sum graph
    def add(self,ax,ay):
        self.alignedSource()
        # Trasforming in two matrix
        x=self.aX.matrix()
        y=self.aY.matrix()
        adjX=self.aX.adj
        adjY=self.aY.adj
        # Nodes
        nX=self.aX.n_nodes
        new={}
        # new set of nodes
        fullset=set(x.keys()).union(set(y.keys()))
        for i in range(nX):
            # node in both networks
            if((i,i) in x and (i,i) in y):
                new[i,i]=self.summ(ax,x[i,i],ay,y[i,i])
            # node in one of the two
            elif((i,i) in x and not (i,i) in y):
                new[i,i]=self.summ(ax,x[i,i],ay,None)
            elif((not (i,i) in x) and (i,i) in y):
                new[i,i]=self.summ(ax,None,ay,y[i,i])
            linked_nodes=[]
            if(i in adjX and i in adjY):
                linked_nodes=set(adjX[i]).union(set(adjY[i]))
            else:
                if(i in adjX and not i in adjY):
                    linked_nodes=set(adjX[i])
                if(i in adjY and not i in adjX):
                    linked_nodes=set(adjY[i])
                    
            for j in linked_nodes:
                # edge doesn't exist in both networks (impossible)
                if((not (i,j) in y) and (not (i,j) in x)):
                       continue
                # edge exists in both networks
                elif((i,j) in y and (i,j) in x):
                    new[i,j]=self.summ(ax,x[i,j],ay,y[i,j])
                # edge exists in one of the two
                elif(not (i,j) in y):
                    new[i,j]=self.summ(ax,x[i,j],ay,None)
                elif(not (i,j) in x):
                    new[i,j]=self.summ(ax,None,ay,y[i,j])
            newG=Graph(x=new,s=self.Y.s,adj=None)
        return newG
    
