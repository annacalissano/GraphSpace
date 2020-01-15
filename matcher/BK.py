# package matcher;
from core import Graph
from distance import euclidean
from scipy.sparse import lil_matrix,vstack
import numpy as np
import math
from matcher import Matcher
import copy

# Generalized Bron Kerbosch Algorithm
# author: brijneshjain

# BK is a child of matcher
# BK algorithm is used to compute the match between two network
class BK(Matcher):

    
        
    def startBK(self,X,Y):
        self.opt_maxRecFactor=-1.0
        self.associate(X,Y)
        self.maxRec=round(self.opt_maxRecFactor*self.numNodes)#round(self.opt_maxRecFactor*self.XxY.shape[0])
        self.numRec = 0
        self.optRec = 0
        self.optClq = [0]*self.minNodes
        for i in range(self.minNodes):
            self.optClq[i]=i*self.nY+i
        self.optSim=0.0
        var9=self.optClq
        var2=len(var9)
        for var3 in range(var2):
            i=var9[var3]
            var5=self.optClq
            var6=len(var5)
            for var7 in range(var6):
                j=var5[var7]
                if((i,j) in self.XxY):
                    self.optSim+=self.XxY[i,j]

    def numRecursions(self):
        return self.numRec

    def numOfRecursionsAtOptimim(self):
        return self.optRec
    
    def match(self,X,Y):
        self.startBK(X,Y)
        C=[]
        P=[]#np.empty(self.numNodes)
        S=[]#np.empty(self.numNodes)
        for i in range(self.numNodes):
            P+=[i] # CHECK, maybe is not the same
        self.search(0.0,C,P,S)
        self.setf()
        
        
    # Search function
    def search(self,sim,C,P,S):
        self.numRec+=1
        if(self.maxRec<=0 or self.numRec<=self.maxRec):
            if(len(P)==0):
                if(len(S)==0 and self.optSim<sim):
                    self.optRec=self.numRec
                    self.optSim=sim
                    self.optClq=copy.deepcopy(C)
            else:
                if(sim+self.h(P,C)>=self.optSim): # h function
                    p=self.reorder(P,C) # reorder function
                    var7=p
                    var8=len(p)
                    for var9 in range(var8):
                        xi=var7[var9]
                        P.remove(xi)
                        # function reduce
                        S2=self.reduce(S,xi)
                        P2=self.reduce(P,xi)
                        # function bound
                        if(not self.bound(P2,S2)):
                            # insert function
                            C2=self.insert(C,xi)
                            sim2=self.getSim(sim,C,xi)
                            # Prova:
                            self.sim2_p=sim2
                            self.C2_p=C2
                            self.P2_p=P2
                            self.S2_p=S2
                            
                            self.search(sim2,C2,P2,S2)
                            S+=[xi]

    def reduce(self,L,x):
        nl=len(L)
        L2=[]#np.empty(nl)
        for i in range(nl):
            j=L[i]
            if(x!=j and (x,j) in self.XxY): # self.a[i,j]
                L2+=[j]
        return L2
    
    
    def bound(self,P,S):
        if(len(P)!=0 and len(S)!=0):
            nS=len(S)
            nP=len(P)
            for i in range(nS):
                s=S[i]
                connected=True
                for j in range(nP):
                    p=P[j]
                    if(not (p!=s and (p,s) in self.XxY)): # not self.a[p,s]
                        connected=False
                        break
                if(connected):
                    return True
            return False
        else: return False
    
    
    def getSim(self,simC,C,xi):
        sim=simC+self.XxY[xi,xi]
        var7=C
        var8=len(C)
        for var9 in range(var8):
            x=var7[var9]
            if((x,xi) in self.XxY):
                sim+=self.XxY[x,xi]
            if((xi,x) in self.XxY):
                sim+=self.XxY[xi,x]
        return sim
    
    
    
    def h(self,P,C):
        projPX=self.proj(P,self.pX)
        projPY=self.proj(P,self.pY)
        projCX=self.proj(C,self.pX)
        projCY=self.proj(C,self.pY)
        glen=self.sqLen(projPX,self.x2)
        hlen=self.sqLen(projPY,self.y2)
        glen+=self.sqLen(projPX,projCX,self.x2)
        hlen+=self.sqLen(projPY,projCY,self.y2)
        out=math.sqrt(glen)*math.sqrt(hlen)
        return out
    
    # possible calls for proj function
    def sqLen(self,*args):
        if(len(args)==2):
            proj=args[0]
            self.x2=args[1]
            sqlen=0.0
            n=len(proj)
            for i in range(n):
                proji=proj[i]
                sqlen+=self.x2[proji,proji]
                for j in range(i,n):
                    sqlen+=self.x2[proji,proj[j]]+self.x2[proj[j],proji]
                    
            return sqlen
        if(len(args)==3):
            projP=args[0]
            projC=args[1]
            self.x2=args[2]
            sqlen=0.0
            var6=projP
            var7=len(projP)
            for var8 in range(var7):
                pi=var6[var8]
                var10=projC
                var11=len(projC)
                for var12 in range(var11):
                    cj=var10[var12]
                    sqlen+=self.x2[pi,cj]+self.x2[cj,pi]
            return sqlen

    # Check, proj exist for two version of pnodes: integerarray and list of int
    def proj(self,pnodes,p):
        n=len(pnodes)
        proj=[]#np.empty(n)
        sel=[False]*len(p) # boolean[] sel = new boolean[p.length]
        for i in range(n):
            xi=pnodes[i]
            
            if(not sel[p[xi]]):
                proj+=[p[xi]]
                sel[p[xi]]=True
                  
        return proj        
                    
            


    # CHECK!
    def clone(self):
        return copy.deepcopy(self)
    
    # initialize a paramenter of distance
    def BK_measure(self,distance,maxNumRecursionsFactor):
        self.opt_maxRecFactor = maxNumRecursionsFactor
   
    def insert(self,c,x):
        n=len(c)
        c2=copy.deepcopy(c)
        c2.append(x)
        return c2
    # Create the association matrix to compute the maximal weighted clique of the association
    def associate(self,X,Y):
        self.x=X.matrix()
        self.y=Y.matrix()
        self.nX=X.nodes()
        self.nY=Y.nodes()
        self.x2=self.square(X)
        self.y2=self.square(Y)
        self.minNodes=min(self.nX,self.nY)
        self.numNodes=self.nX*self.nY
        self.XxY={}#lil_matrix((self.numNodes,self.numNodes))
        #self.a={}#lil_matrix((self.numNodes,self.numNodes))
        for i in range(self.nX):
            for j in range(self.nY):
                ij=i*self.nY+j
                # every entry in the diagonal is the distance between nodes
                if((i,i) in self.x and (j,j) in self.y):
                    self.XxY[ij,ij]=self.measure.node_sim(self.x[i,i],self.y[j,j])
                else:
                    if((i,i) in self.x and not (j,j) in self.y):
                        self.XxY[ij,ij]=self.measure.node_sim(self.x[i,i],[])
                    if((not (i,i) in self.x) and (j,j) in self.y):
                        self.XxY[ij,ij]=self.measure.node_sim([],self.y[j,j])
                
                
                
                for k in range(self.nX):
                    for l in range(self.nY):
                        kl=k*self.nY+l
                        if(i!=k and l!=j):
                            #self.a[ij,kl]=True
                            if((i,k) in self.x and (j,l) in self.y):
                                # out of the diagonal we have the distance between edges
                                # only if edges exit on both side
                                self.XxY[ij,kl]=self.measure.edge_sim(self.x[i,k],self.y[j,l])
                            if((i,k) in self.x and not (j,l) in self.y):
                                self.XxY[ij,kl]=self.measure.edge_sim(self.x[i,k],[])
                            if(not (i,k) in self.x and (j,l) in self.y):
                                self.XxY[ij,kl]=self.measure.edge_sim(self.y[j,l],[])
                        #else: self.a[ij,kl]=False
        self.pX=[0]*self.numNodes
        self.pY=[0]*self.numNodes
        for i in range(self.numNodes):
            self.pX[i]=i/self.nY
            self.pY[i]=i%self.nY

        
    # square of a graph
    def square(self,Z):
        n=Z.nodes()
        z=Z.matrix()
        adj=Z.adj
        z2=lil_matrix((n,n))
        for i in range(n):
            if((i,i) in z):
                z2[i,i]=self.measure.node_sim(z[i,i],z[i,i])
            else:
                z2[i,i]=self.measure.node_sim([],[])
            deg=Z.degree(i)
            for j in range(deg):
                j0=adj[i][j]
                z2[i,j0]=self.measure.edge_sim(z[i,j0],z[i,j0])
        return z2

     # reorder
    def reorder(self, P,C):
        nP=len(P)
        sim=np.empty(nP)
        maxS = -1.7976931348623157e+308
        x0=0
        for i in range(nP):
            #xi=P[i]
            sim[i]=self.XxY[P[i],P[i]]
            #var10=C
            len_c=len(C)
            for j in range(len_c):
                #v_j=var10[j]
                if((j,C[j]) in self.XxY):
                    sim[i]+=self.XxY[j,C[j]]
                if((C[j],j) in self.XxY):
                    sim[i]+=self.XxY[C[j],j]
            
            if(sim[i]>maxS):
                maxS=sim[i]
                x0=P[i]
        
        P2=[0]*nP
        sim2=[0.0]*nP
        nP2=0
        for i in range(nP):
            #xi=P[i]
            if(not (x0!=P[i] and (x0,P[i]) in self.XxY)):
                P2[nP2]=P[i]
                sim2[nP2]=-sim[i]
                nP2+=1
        f=sorted(range(nP2), key=lambda k: sim2[k])
        p2=[0]*nP2
        for i in range(nP2):
            p2[i]=P2[f[i]]
        return p2
        
    def setf(self):
        self.f=[-1]*self.nX
        var1=self.optClq
        i=len(var1)
        for var3 in range(i):
            i=var1[var3]
            self.f[self.pX[i]]=self.pY[i]
        if(self.nY<self.nX):
            c=self.nY
            for i in range(self.nX):
                if(self.f[i]<0):
                    self.f[i]=c
                    c+=1