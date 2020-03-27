# The following class run the following algorithm approach:
# Take: X1,..,Xn obstervation in our Q quotient space
# Step1: Initialize Observation in m=Xi
# Step2: Align X2,...,Xn wrt m (save the permutation vector)
# Step3: Compute the generalized geodesic regression (ggr) on the total space of aligned graphs
# Step4: save the coefficient of the regression
# Step5: compute the sum_residuals as a measurment of the ggr quality
# repreat step2-4 until the sum_residuals is not changing

from core import Graph
from core import GraphSet
from AlignCompute import aligncompute
import numpy as np
import copy
import math
import re
import pandas as pd
from sklearn import linear_model
import random
from matcher import GA,ID
import scipy

class ggr_aac(aligncompute):
    
    def __init__(self,graphset,matcher,distance):
        aligncompute.__init__(self,graphset,matcher)
        self.mean=None
        self.measure=distance
        # Compute the domain [s_min,s_max] along which we are aligning with respect to
        # the geodesic
        self.step=[]
        self.step_coef=GraphSet()
        # the regression errors after alignment
        self.error=[]
        self.regression_error=pd.DataFrame(index=range(graphset.size()), columns=range(100))
        self.postalignment_error = pd.DataFrame(index=range(graphset.size()), columns=range(100))

    def align_and_est(self):
        # INITIALIZATION:
        # Select a Random Candidate:
        first_id = random.randint(0, self.aX.size() - 1)
        m_1 = self.aX.X[first_id]
        self.f[first_id] = range(self.aX.n_nodes)
        # Align all the points wrt the random candidate
        for i in range(self.X.size()):
           # Align X to Y
           a = self.matcher.align(self.aX.X[i], m_1)
           # Permutation of X to go closer to Y
           self.f[i] = a.f


        # Compute the first Generalized Geodesic Regression line
        E_1 = self.est(k=0)
        # Align the set wrt the geodesic
        self.align_pred(E_1[1],k=0)
        self.error += [sum(E_1[0]._residues)]
        # AAC iterative algorithm
        # k=200 maximum number of iteration
        for k in range(1,20):
            # Compute the first Generalized Geodesic Regression line
            E_2 = self.est(k)
            # Align the set wrt the geodesic
            self.align_pred(E_2[1],k)
            # Compute the step: the algorithmic step is computed as the square difference between the coefficients
            step_range = abs(sum(E_1[0]._residues)-sum(E_2[0]._residues))
            self.error+=[sum(E_2[0]._residues)]

            self.step+=[step_range]
            self.step_coef.add(self.give_me_a_network(pd.Series(data=E_2[0].coef_.flatten(),index=self.variables_names),self.aX.node_attr,self.aX.edge_attr))
            # if(step_range<0.000005):
            #     #IF small enough, I am converging! Save and exit.
            #     self.network_coef = GraphSet()
            #     # self.vector_coef = pd.Series(data=E_2[0].coef_.flatten(), index=self.variables_names)
            #     self.network_coef.add(
            #         self.give_me_a_network(pd.Series(data=E_2[0].intercept_.flatten(), index=self.variables_names),
            #                                self.aX.node_attr, self.aX.edge_attr, y='Intercept'))
            #     for i_th in range(E_2[0].coef_.shape[1]):
            #         self.network_coef.add(
            #             self.give_me_a_network(pd.Series(data=E_2[0].coef_[:, i_th], index=self.variables_names),
            #                                    self.aX.node_attr, self.aX.edge_attr, y=str('beta' + str(i_th))))
            #     self.model = E_2[0]
            #     self.sum_residuals = sum(E_2[0]._residues)
            #     print("Step Range smaller than 0.005")
            #     return
            #else Go on with the computation: update the new result and restart from step 1.
            del E_1
            E_1=E_2
            del E_2
        print("Maximum number of iteration reached.")  
        # Return the result
        if('E_2' in locals()):
            self.network_coef =GraphSet()
            #self.vector_coef = pd.Series(data=E_2[0].coef_.flatten(), index=self.variables_names)
            self.network_coef.add(self.give_me_a_network(pd.Series(data=E_2[0].intercept_.flatten(), index=self.variables_names), self.aX.node_attr, self.aX.edge_attr,y='Intercept'))
            for i_th in range(E_2[0].coef_.shape[1]):
                self.network_coef.add(
                    self.give_me_a_network(pd.Series(data=E_2[0].coef_[:,i_th], index=self.variables_names),
                                           self.aX.node_attr, self.aX.edge_attr, y=str('beta'+str(i_th))))
            self.model = E_2[0]
            self.sum_residuals = sum(E_2[0]._residues)
            del E_2,E_1
        else:
            self.network_coef =GraphSet()
            #self.vector_coef = pd.Series(data=E_2[0].coef_.flatten(), index=self.variables_names)
            self.network_coef.add(self.give_me_a_network(pd.Series(data=E_1[0].intercept_.flatten(), index=self.variables_names), self.aX.node_attr, self.aX.edge_attr,y='Intercept'))
            for i_th in range(E_1[0].coef_.shape[1]):
                self.network_coef.add(
                    self.give_me_a_network(pd.Series(data=E_1[0].coef_[:,i_th], index=self.variables_names),
                                           self.aX.node_attr, self.aX.edge_attr, y=str('beta'+str(i_th))))
            self.model = E_1[0]
            self.sum_residuals = sum(E_1[0]._residues)
            del E_1
        
        
    # Align wrt a geodesic
    def align_pred(self,y_pred,k):# delete k, only for the error test
        self.f.clear()
        # the alignment wrt a geodesic aiming at predicting data is an alignment wrt the prediction along
        # the regression gamma(x_i) and the data point itself y_i
        # i.e. find the optimal candidate y* in [y] st d(gamma(x)-y) is minimum
        self.aX.get_node_attr()
        self.aX.get_edge_attr()
        #print("The network ith to aling")
        #print(self.aX.to_matrix_with_attr())
        #print("The predicted network")
        #print(y_pred)
        # for every graph save the new alignment
        for i in range(self.aX.size()):
            # transform the estimation into a network to compute the networks distances
            y_pred_net= self.give_me_a_network(y_pred.iloc[i], self.aX.node_attr, self.aX.edge_attr)
            #a = self.matcher.align(y_pred_net,self.aX.X[i])
            #self.f[i] = a.f
            self.postalignment_error.iloc[i,k]=self.matcher.dis(y_pred_net, self.aX.X[i])
            self.f[i] = self.matcher.f
            #self.postalignment_error.iloc[i,k]=a.dis()

            del(y_pred_net)

    # Compute the generalized geodesic regression on the total space as a regression of the aligned graph set
    def est(self,k): # delete k after the error check
        # dimension of the dataset
        N=self.aX.size()
        # Step 1: Create the current permuted dataset
        G_per=GraphSet()
        for i in range(N):
            G_temp=copy.deepcopy(self.aX.X[i])
            G_temp.permute(self.f[i])
            G_temp.y=copy.deepcopy(self.aX.X[i].y)
            G_per.add(G_temp)
            del(G_temp)
        del(self.aX)
        self.aX=copy.deepcopy(G_per)
        # Transform it into a matrix
        y = G_per.to_matrix_with_attr()
        # parameter saved:
        self.variables_names=y.columns
        # Create the input value
        t = []
        for i in range(y.shape[0]):
            t += [float(G_per.X[i].y)]
        x = pd.DataFrame(data=t, index=y.index)
        #print("The Y to fit")
        #print(y)
        #print("The x to fit")
        #print(x)
        # Create linear regression object
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        along_geo_pred = pd.DataFrame(regr.predict(x),columns=self.variables_names)
        self.regression_error.iloc[:,k] = (along_geo_pred-y).pow(2).sum(axis=1)
        return (regr,along_geo_pred)

    # Given x_new is predicting the corresponding graph:
    def predict(self,x_new):
        if(not isinstance(x_new,pd.core.frame.DataFrame)):
            print("The new observation should be a pandas dataframe of real values")
        self.y_vec_pred=self.model.predict(X=x_new)
        self.y_net_pred=GraphSet()
        for i in range(self.y_vec_pred.shape[0]):
            self.y_net_pred.add(self.give_me_a_network(geo=pd.Series(data=self.y_vec_pred[i],index=self.variables_names),n_a=self.aX.node_attr,e_a=self.aX.edge_attr,y=float(x_new.loc[i])))

#    def save(self,filename):




    # These functions are auxiliary function to compute the ggr
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
    
    # component wise distance function: usefull to compute the covariance
    def dis_componentwise(self,A,B,f):
        # Adjency Matrix: x, y
        y=B.x
        G=copy.deepcopy(A)
        G.permute(f)
        x=G.x
        # coefficients: ax, ay
        # Links
        adjX=G.adj
        adjY=B.adj
        nX=A.n_nodes
        new={}
        fullset=set(x.keys()).union(set(y.keys()))
        for i in range(nX):
            if((i,i) in x and (i,i) in y):
                new[i,i]=[math.sqrt(self.measure.node_dis(x[i,i],y[i,i]))]
            elif((i,i) in x and not (i,i) in y):
                new[i,i]=[math.sqrt(self.measure.node_dis(x[i,i],[0]))]
            elif((not (i,i) in x) and (i,i) in y):
                new[i,i]=[math.sqrt(self.measure.node_dis(y[i,i],[0]))]
            
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
                    new[i,j]=[math.sqrt(self.measure.edge_dis(x[i,j],y[i,j]))]
                elif(not (i,j) in y):
                    new[i,j]=[math.sqrt(self.measure.edge_dis(x[i,j],[0]))]
                elif(not (i,j) in x):
                    new[i,j]=[math.sqrt(self.measure.edge_dis([0],y[i,j]))]
        newG=Graph(x=new,y=None,adj=None)
        return newG
    
    # geo is a pd Series
    # n_a and e_a are nodes and edges attributes
    def give_me_a_network(self,geo,n_a,e_a,y=None):
        ind=[re.findall(r'-?\d+\.?\d*', k) for k in geo.axes[0]]
        x_g={}
        for i in range(len(ind)):
            if(len(ind[i])>2 and int(ind[i][0])==int(ind[i][1]) and not (int(ind[i][0]),int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]),int(ind[i][1])]=[geo.loc[geo.axes[0][i+j]] for j in range(n_a)]
            elif(len(ind[i])>2 and int(ind[i][0])!=int(ind[i][1]) and not (int(ind[i][0]),int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]),int(ind[i][1])]=[geo.loc[geo.axes[0][i+j]] for j in range(e_a)]
            elif(len(ind[i])==2 and not (int(ind[i][0]),int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]),int(ind[i][1])]=[geo.loc[geo.axes[0][i]]]
        
        geo_net=Graph(x=x_g,adj=None,y=y)
        return geo_net
