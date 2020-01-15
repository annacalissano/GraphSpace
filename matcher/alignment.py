from core import Graph
from distance import euclidean
import numpy as np
import copy

class alignment:
    
    # We Align X to Y with f permutation of Y
    def __init__(self,X,Y,f,measure):
        self.X=X # to align original graph
        self.Y=Y #target original graph
        self.aY=copy.deepcopy(Y) #to aligned y # ATTENTION: it is creating a copy .cp() function of it
        self.aY=self.aY.grow(self.X.nodes()) #aligned y
        self.aX=None # to aligned y
        self.f=f # permuting vector
        self.measure=measure # the distance between networks
        #print('Buddy remember we are aligning to Y, so f should be a permutation of Y (i.e. the biggest matrix)')

    
    
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
    
    # aligning: expand and permute
    def alignedSource(self):
        if(self.aX is None):
            self.aX=copy.deepcopy(self.X)
            self.aX.grow(self.Y.nodes())
            self.aX.permute(self.f)
        #return self.aX
    
    # Define or return the alignment source
    def alignedTarget(self):
        return self.aY
    
    # Compute the distance:
    def dis(self):
        self.alignedSource()
        nX=self.aX.nodes()
        dis=0.0
        # the two adjency matrix
        x=self.aX.matrix()
        y=self.aY.matrix()
        adjX=self.aX.adj
        adjY=self.aY.adj
        for i in range(nX):
            
            
            if((i,i) in x and (i,i) in y):
                dis+=self.measure.node_dis(x[i,i],y[i,i])
            elif((i,i) in x and not (i,i) in y):
                dis+=self.measure.node_dis(x[i,i],[0])
            elif((not (i,i) in x) and (i,i) in y):
                dis+=self.measure.node_dis([0],y[i,i])

            linked_nodes=[]
            if(i in adjX and i in adjY):
                linked_nodes=set(adjX[i]).union(set(adjY[i]))
            else:
                if(i in adjX and not i in adjY):
                    linked_nodes=set(adjX[i])
                if(i in adjY and not i in adjX):
                    linked_nodes=set(adjY[i])
                    
            for j in linked_nodes:
                # Both edges don't exist in both networks (impossible)
                if((not (i,j) in y) and (not (i,j) in x)):
                       continue
                # Both edges exist in both networks
                elif((i,j) in y and (i,j) in x):
                    dis+=self.measure.edge_dis(x[i,j],y[i,j])
                elif(not (i,j) in y):
                    dis+=self.measure.edge_dis(x[i,j],[0])
                elif(not (i,j) in x):
                    dis+=self.measure.edge_dis([0],y[i,j])
        return dis
          
    # Sim function for the networks
    def sim(self):
        nX=self.X.nodes()
        # the two adjency matrix
        #self.alignedSource()
        # sure? ay and X?
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
       
    
    
    # add function:
    def add(self,ax,ay):
        self.alignedSource()
        x=self.aX.matrix()
        y=self.aY.matrix()
        #print 'sum:'
        #print ax
        #print x
        #print ay
        #print y
        # Links
        adjX=self.aX.adj
        adjY=self.aY.adj
        # Nodes
        nX=self.aX.n_nodes
        new={}
        fullset=set(x.keys()).union(set(y.keys()))
        for i in range(nX):
            
            
            if((i,i) in x and (i,i) in y):
                new[i,i]=self.summ(ax,x[i,i],ay,y[i,i])
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
                # Both edges don't exist in both networks (impossible)
                if((not (i,j) in y) and (not (i,j) in x)):
                       continue
                # Both edges exist in both networks
                elif((i,j) in y and (i,j) in x):
                    new[i,j]=self.summ(ax,x[i,j],ay,y[i,j])
                elif(not (i,j) in y):
                    new[i,j]=self.summ(ax,x[i,j],ay,None)
                elif(not (i,j) in x):
                    new[i,j]=self.summ(ax,None,ay,y[i,j])
            newG=Graph(x=new,y=self.Y.y,adj=None)
        return newG
    