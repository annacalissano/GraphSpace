from core import Graph
from distance import euclidean
from scipy.sparse import lil_matrix,vstack
import numpy as np
import math
from munkres import Munkres
from matcher import Matcher
from scipy import sparse

# Graduate assigned algoritm
# author: brijneshjain

# GA is a child of matcher
# GA algorithm is used to compute the match between two network
class GA(Matcher):

    global opt_b0,opt_bf,opt_br,opt_I0,opt_I1,opt_eps0,opt_eps1
    opt_b0 = 0.5
    opt_bf = 10.0
    opt_br = 1.075

    # max # of iterations of inner and outer loop
    opt_I0 = 4;
    opt_I1 = 30;

    # precision for terminantion in inner and outer loop
    opt_eps0 = 0.5;
    opt_eps1 = 0.05;
    # **********************

    #// continuation parameter
    #private double b;

    
    #// match matrix
    #private double[][] M;

    #// association graph
    #private double[][][][] A;
    #private double[][] a;

    #// flag, indicating whether argument has been swapped for better performance
    #private boolean swapped;

    def __init__(self,X=None,Y=None,f=None):
        Matcher.__init__(self,X,Y,f)
        self.a=None
        self.A=None
        self.M=None
        self.b=opt_b0
        self.swapped=False
        self.name="Graduate assignement"
        
    
    # The match function: this function find the best match (max or min) among the equivalent classes
    def match(self,X,Y):
        # Take the two graphs
        self.X=X
        self.Y=Y
        # check the dimensions and set everything to start
        if(self.X.nodes()>self.Y.nodes()):
            # swap the two network
            self.swap()
            self.swapped=True
        # initialize match
        self.initializeMatchMatrix()
        self.setAssociationGraph()
        
        nX=self.X.nodes()
        nY=self.Y.nodes()

        adjX=self.X.adj
        adjY=self.Y.adj

        # new parameters
        # Partial derivative matrix taylor expansion
        Q=lil_matrix((nX, nY))
        
        # QUI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # M0 exp equation
        M0=lil_matrix((nX+ 1, nY+ 1))
        #M0=lil_matrix((nX, nY))
        # M1 scaled M0
        M1=lil_matrix((nX+ 1, nY+ 1))
        #M1=lil_matrix((nX, nY))

        # A loop
        self.b = opt_b0
        while (self.b < opt_bf):
            
            # B loop
            for  t0 in range(opt_I0):
                # copy
                # QUI !!!!!!!!!!!!!!!!!!!!!!!!!!!
                for i in range(nX+1):
                #for i in range(nX):
                    M0[i,]=self.M[i,0:nY+1] #System.arraycopy(M[i], 0, M0[i], 0, nY + 1);
                    #M0[i,]=self.M[i,0:nY]

                # softmax
                for i in range(nX):
                    for j in range(nY):
                        Q[i,j]=self.a[i,j]
                        degX=self.X.degree(i)
                        for k in range(degX):
                            degY=self.Y.degree(j)
                            if(degY!=0):
                                for l in range(degY):
                                    # equation
                                    Q[i,j]+=self.A[i,j][k,l]*M0[adjX[i][k],adjY[j][l]]
                        self.M[i,j]=math.exp(self.b*Q[i,j])
                ## C loop
                for t1 in range(opt_I1):
                    
                    # QUI !!!!!!!!!!!!!!!!!!!!!
                    #for i in range(nX):
                    for i in range(nX+1):
                        # copy
                        M1[i,]=self.M[i,0:nY+1]
                        #M1[i,]=self.M[i,0:nY]
                    # normalize across all rows
                    for i in range(nX+1):
                    #for i in range(nX):
                        row_sum=0
                        for j in range(nY+1):
                            #for j in range(nY):
                            row_sum+=self.M[i,j]
                        for j in range(nY+1):
                            #for j in range(nY):
                            self.M[i,j]/=row_sum
                    # normalize across all columns
                    for j in range(nY+1):
                        #for j in range(nY):
                        col_sum=0
                        
                        for i in range(nX+1):
                            #for i in range(nX):
                            col_sum+=self.M[i,j]
                        for i in range(nX+1):
                            #for i in range(nX):
                            self.M[i,j]/=col_sum
                   
                    # check for convergence
                    if(self.isStable(self.M,M1,opt_eps1)):
                        break
                # end C loop
                 
                if(self.isStable(self.M,M0,opt_eps0)):
                       break
            # end B loop
            self.b *= opt_br;
                   
        # end A loop
        self.cleanup()

            
    # set parameters function
    def initializeMatchMatrix(self):
        nX=self.X.nodes()
        nY=self.Y.nodes()
        # Matching matrix variable
        self.M=np.full((nX+1, nY+1),1.001)
   
    # compute and initialize the self.a and self.A
    def setAssociationGraph(self):
        x=self.X.matrix()
        y=self.Y.matrix()
        adjX=self.X.adj
        adjY=self.Y.adj
        scale=0
        nX=self.X.nodes()
        nY=self.Y.nodes()
        self.A={} # distance edges
        self.a=np.empty([nX,nY]) # distance nodes
        for i in range(nX):
            degX=self.X.degree(i)
            for j in range(nY): 
                
                degY=self.Y.degree(j)
                # node distance
                self.a[i,j]=self.measure.node_sim(x[i,i],y[j,j])
                scale=max(abs(self.a[i,j]),scale)
                self.A[i,j]=lil_matrix((degX, degY))
                for k in range(degX):                        
                    k0=adjX[i][k]
                    for l in range(degY):
                        l0=adjY[j][l]
                        self.A[i,j][k,l]=self.measure.edge_sim(x[i,k0],y[j,l0])
                        scale=max(abs(self.A[i,j][k,l]),scale)
        if(scale==0):
            return self
        for i in range(nX):
            for j in range(nY):
                self.a[i,j]=self.a[i,j]/scale
                degX=self.X.degree(i)
                for k in range(degX):
                    degY=self.Y.degree(j)
                    for l in range(degY):
                        self.A[i,j][k,l]=self.A[i,j][k,l]/scale

    # computing the matching with the hungarian algorithm                    
    def cleanup(self):
        nX=self.X.nodes()
        nY=self.Y.nodes()
        C=np.empty([nX,nY])
        for i in range(nX):
            for j in range(nY):
                C[i,j]=1.0-self.M[i,j]
        #C=1.0-self.M
        #C_reduce = lil_matrix(sparse.csr_matrix(C)[0:nX,0:nX]).todense()
        m=Munkres()
        indexmatch=m.compute(C)
        self.f=[indexmatch[i][1] for i in range(len(indexmatch))]
        self.f=self.f[0:nX]
        #f=Munkres.computeAssignments(C,10E6)
        if(self.swapped):
            g=[]
            for i in range(nY):
                g[f[i]]=i
            self.f=g
            self.swap(self)
   

    # swapping the two
    def swap(self):
        tmp=self.X
        self.X=self.Y
        self.Y=tmp
    
    # check how the algprithm is going
    def isStable(self,M1,M2,eps):
        nX=self.X.nodes()
        nY=self.Y.nodes()
        err=np.sum(abs(M1-M2))
        err/=nX*nY
        return (err<eps)

