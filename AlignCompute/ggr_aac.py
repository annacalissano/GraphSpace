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
from distance import sqeuclidean
import copy
import re
import pandas as pd
from sklearn import linear_model,gaussian_process
from joblib import Parallel,delayed
import random
from matcher import GA,ID
import scipy
from joblib import Parallel, delayed

class ggr_aac(aligncompute):
    
    def __init__(self,graphset,matcher,distance,regression_model='OLS',nr_iterations=None):
        # distance and matcher used to compute the alignment
        aligncompute.__init__(self,graphset,matcher)
        # distance used to compute the regression error
        self.distance=distance
        # nr of iteration of the algorithm
        if(nr_iterations==None):
            self.nr_iterations=100
        else:
            self.nr_iterations = nr_iterations
        # indicate which type of regression model:
        # OLS (e.g. network on scalar regression problems)
        # GPR (e.g. network on time regression problems)
        self.model_type=regression_model
        # if (self.model_type=='GPR'):
        #     self.alpha=alpha
        #     self.restarts=restarts
        #     self.models={}
        #     if(kernel == None):
        #         # by deafault we select an exponential kernel
        #         # See kernel section in gaussian_process documentation
        #         # https://scikit-learn.org/stable/modules/gaussian_process.html#gp-kernels
        #         # Here we used: 1/2exp(-d(x1/l,x2/l)^2)
        #         # - s is the parameter of the ConstantKernel
        #         # - l is the parameter of the RBF (radial basis function) kernel
        #         self.kernel = gaussian_process.kernels.ConstantKernel(1.0) * gaussian_process.kernels.RBF(1.0)
        #     else:
        #         self.kernel=kernel
        # # Regression error for each iteration and each observation
        self.regression_error={}#pd.DataFrame(0,index=range(graphset.size()), columns=range(self.nr_iterations))
        self.postalignment_error = {}#pd.DataFrame(0,index=range(graphset.size()), columns=range(self.nr_iterations))
        self.f_iteration={}
        self.f_all = {}

    def align_and_est(self):
        # INITIALIZATION:
        # Select a Random Candidate:
        first_id = 0#random.randint(0, self.aX.size() - 1)
        m_1 = self.aX.X[first_id]
        while(m_1.n_nodes==1):
            first_id = random.randint(0, self.aX.size() - 1)
            print(first_id)
            m_1 = self.aX.X[first_id]
            print(m_1.x)
        # # Sequential version:
        # if(self.parallel=False):
        #     # Align all the points wrt the random candidate
        #     for i in range(self.X.size()):
        #         # Align X to Y
        #         a = self.matcher.dis(self.aX.X[i],m_1)
        #         # Permutation of X to go closer to Y
        #         self.f[i] = self.matcher.f
        # # Parallel Version:
        # if(self.parallel=True):

        Parallel(n_jobs=1, verbose=1,require='sharedmem')(
            delayed(self.two_net_match)(m_1, i,first_id) for i in range(self.aX.size()))
        # Compute the first Generalized Geodesic Regression line
        E_1 = self.est(k=0)
        # Align the set wrt the geodesic
        Parallel(n_jobs=1, verbose=1, require='sharedmem')(
                delayed(self.align_pred)(E_1[1], i,0) for i in range(self.aX.size()))
        # AAC iterative algorithm
        for k in range(1,self.nr_iterations):
            # Compute the first Generalized Geodesic Regression line
            E_2 = self.est(k)
            # Align the set wrt the geodesic
            Parallel(n_jobs=1, verbose=1, require='sharedmem')(
                delayed(self.align_pred)(E_2[1], i,k) for i in range(self.aX.size()))
            #sequential version: self.align_pred(E_2[1],k)
            # Compute the step: the algorithmic step is computed as the square difference between the coefficients
            step_range = abs(sum([self.regression_error[i,k-1] for i in range(0,self.aX.size())])-sum([self.regression_error[i,k] for i in range(0,self.aX.size())]))
            #self.error+=[self.regression_error.iloc[:,k].sum()]

            if(step_range<0.05):
                self.model = E_2[0]
                if (self.model_type == 'OLS' or self.model_type=='OLS_Anova'):
                    # Return the coefficients
                    self.network_coef = GraphSet()
                    # self.vector_coef = pd.Series(data=E_2[0].coef_.flatten(), index=self.variables_names)
                    self.network_coef.add(
                        self.give_me_a_network(pd.Series(data=E_2[0].intercept_.flatten(), index=self.variables_names),
                                               self.aX.node_attr, self.aX.edge_attr, s='Intercept'))
                    for i_th in range(E_2[0].coef_.shape[1]):
                        self.network_coef.add(
                            self.give_me_a_network(pd.Series(data=E_2[0].coef_[:, i_th], index=self.variables_names),
                                                   self.aX.node_attr, self.aX.edge_attr, s=str('beta' + str(i_th))))
                    self.regression_error=pd.DataFrame.from_dict({iteration:[
                        self.regression_error[observation,iteration] for observation in range(self.aX.size())] for
                        iteration in range(k+1)})
                    self.postalignment_error = pd.DataFrame.from_dict({iteration: [
                        self.postalignment_error[observation, iteration] for observation in range(self.aX.size())] for
                                                                    iteration in range(k+1)})
                    self.nr_iterations=k
                print("Step Range smaller than 0.005")
                return
            #else Go on with the computation: update the new result and restart from step 1.
            del E_1
            E_1=E_2
            del E_2
        print("Maximum number of iteration reached.")  
        # Return the result
        if('E_2' in locals()):
            self.model = E_2[0]
            if(self.model_type=='OLS' or self.model_type=='OLS_Anova'):
                # Return the coefficients
                self.network_coef =GraphSet()
                #self.vector_coef = pd.Series(data=E_2[0].coef_.flatten(), index=self.variables_names)
                self.network_coef.add(self.give_me_a_network(pd.Series(data=E_2[0].intercept_.flatten(), index=self.variables_names), self.aX.node_attr, self.aX.edge_attr,s='Intercept'))
                for i_th in range(E_2[0].coef_.shape[1]):
                    self.network_coef.add(
                        self.give_me_a_network(pd.Series(data=E_2[0].coef_[:,i_th], index=self.variables_names),
                                           self.aX.node_attr, self.aX.edge_attr, s=str('beta'+str(i_th))))
                self.regression_error = pd.DataFrame.from_dict(
                    {iteration: [self.regression_error[observation, iteration] for observation in range(self.aX.size())]
                     for iteration in range(self.nr_iterations)})
                self.postalignment_error = pd.DataFrame.from_dict({iteration: [
                    self.postalignment_error[observation, iteration] for observation in range(self.aX.size())] for
                    iteration in range(self.nr_iterations)})
            else:
                # Return the prior and the posterior
                # ATTENTION: CHECK ON THE PRIOR WITH AASA
                self.y_post=E_2[1]
                self.y_post_std=E_2[2]


            del E_2,E_1

        else:
            self.model = E_1[0]
            if(self.model_type=='OLS' or self.model_type=='OLS_Anova'):
                # Return the coefficients
                self.network_coef =GraphSet()
                #self.vector_coef = pd.Series(data=E_2[0].coef_.flatten(), index=self.variables_names)
                self.network_coef.add(self.give_me_a_network(pd.Series(data=E_1[0].intercept_.flatten(), index=self.variables_names), self.aX.node_attr, self.aX.edge_attr,s='Intercept'))
                for i_th in range(E_1[0].coef_.shape[1]):
                    self.network_coef.add(
                        self.give_me_a_network(pd.Series(data=E_1[0].coef_[:,i_th], index=self.variables_names),
                                           self.aX.node_attr, self.aX.edge_attr, s=str('beta'+str(i_th))))
                self.regression_error = pd.DataFrame.from_dict(
                    {iteration: [self.regression_error[observation, iteration] for observation in range(self.aX.size())]
                     for iteration in range(self.nr_iterations)})
                self.postalignment_error = pd.DataFrame.from_dict({iteration: [
                    self.postalignment_error[observation, iteration] for observation in range(self.aX.size())] for
                    iteration in range(self.nr_iterations)})
            else:
                # Return the prior and the posterior
                # ATTENTION: CHECK ON THE PRIOR WITH AASA
                self.y_post=E_1[1]
                self.y_post_std=E_1[2]
            del E_1
        
    # Align wrt a geodesic
    def align_pred(self,y_pred,i,k):
        #self.f.clear()
        # the alignment wrt a geodesic aiming at predicting data is an alignment wrt the prediction along
        # the regression gamma(x_i) and the data point itself y_i
        # i.e. find the optimal candidate y* in [y] st d(gamma(x)-y) is minimum
        self.aX.get_node_attr()
        self.aX.get_edge_attr()
        # Sequential Version:
        # for every graph save the new alignment
        # for i in range(self.aX.size()):
        #     # transform the estimation into a network to compute the networks distances
        #     y_pred_net= self.give_me_a_network(y_pred.iloc[i], self.aX.node_attr, self.aX.edge_attr)
        #     # Regression error:
        #     match=ID(self.distance)
        #     self.regression_error.iloc[i,k]=match.dis(self.aX.X[i],y_pred_net)
        #     # sum of squares of distances
        #     self.postalignment_error.iloc[i,k]=self.matcher.dis(self.aX.X[i],y_pred_net)
        #     self.f[i] = self.matcher.f
        #     del(y_pred_net,match)
        # Parallel Version: see the function at the end of the code
        # transform the estimation into a network to compute the networks distances
        y_pred_net = self.give_me_a_network(y_pred.iloc[i], self.aX.node_attr, self.aX.edge_attr)
        # Regression error:
        match = ID(self.distance)
        self.regression_error[i, k] = match.dis(self.aX.X[i], y_pred_net)
        print("regression error" + str(i))
        self.postalignment_error[i, k] = self.matcher.dis(self.aX.X[i], y_pred_net)
        self.f[i] = self.matcher.f
        print("alignment error" + str(i))
        del (y_pred_net, match)

    # Compute the generalized geodesic regression on the total space as a regression of the aligned graph set
    def est(self,k):
        # Step 1: Create the current permuted dataset
        self.f_iteration[k]=self.f
        G_per=GraphSet()
        for i in range(self.aX.size()):
            G_temp=copy.deepcopy(self.aX.X[i])
            G_temp.permute(self.f[i])
            G_temp.s=copy.deepcopy(self.aX.X[i].s)
            G_per.add(G_temp)
            del(G_temp)
        del(self.aX)
        self.aX=copy.deepcopy(G_per)
        print("Aligned")
        # Step 2: Transform it into a matrix
        y = G_per.to_matrix_with_attr()
        # parameter saved:
        self.variables_names=y.columns

        # Step 3: create the x vector
        # Create the input value
        x = pd.DataFrame(columns=range(len(G_per.X[0].s)), index=range(y.shape[0]))
        for i in range(y.shape[0]):
            x.iloc[i] = [float(regressor) for regressor in G_per.X[i].s]
        self.regressor=x
        # Step 4: fit the chosen regression model
        # Ordinary Least Square
        if(self.model_type=='OLS'):
            # Create linear regression object
            model = linear_model.LinearRegression()
            model.fit(x, y)
            along_geo_pred = pd.DataFrame(model.predict(x),columns=self.variables_names)
            self.f_all[k] = self.f
            #self.regression_error.iloc[:, k] = (along_geo_pred - y).pow(2).sum(axis=1)
            return (model, along_geo_pred)
        elif(self.model_type=='OLS_Anova'):
            # Create linear regression object with dummy variable as input
            model = linear_model.LinearRegression()
            x_dummy= pd.get_dummies(data=x, drop_first=True)
            model.fit(x_dummy, y)
            along_geo_pred = pd.DataFrame(model.predict(x_dummy),columns=self.variables_names)
            self.f_all[k] = self.f
            print("Estimated")
            #self.regression_error.iloc[:, k] = (along_geo_pred - y).pow(2).sum(axis=1)
            return (model, along_geo_pred)

        # # Gaussian Process
        # elif(self.model_type=='GPR'):
        #
        #     along_geo_pred=pd.DataFrame(index=range(y.shape[0]), columns=self.variables_names)
        #     along_geo_pred_sd = pd.DataFrame(index=range(y.shape[0]), columns=self.variables_names)
        #     # list in which we save the temporary regression error
        #     regression_error_temp = []
        #     # We are fitting a different Gaussian process for every variable (i.e. for every node or edge)
        #     for m in range(len(self.variables_names)):
        #         # Inizialize the gaussian process
        #         model = gaussian_process.GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.restarts, alpha=self.alpha)
        #         # Fitting the Gaussian Process means finding the correct hyperparameters
        #         model.fit(x, y.iloc[:,m])
        #         # Saving the model
        #         self.models[self.variables_names[m]]=model
        #         # Predict to compute the regression error (to compare with the alignment error)
        #         y_pred,y_std=model.predict(x,return_std=True)
        #         # save both the predicted y and the std, to estimate the posterior
        #         along_geo_pred.loc[:,self.variables_names[m]] = pd.Series(y_pred)
        #         along_geo_pred_sd.loc[:, self.variables_names[m]] = pd.Series(y_std)
        #         # Compute the error
        #         # HERE! YOU CAN SUBSTITUTE IT WITH AN ERROR FUNCTION
        #         err_euclidean = (y_tr.iloc[:, 2] - y_pred).pow(2)
        #         err_weighted=[err_euclidean[i] / y_std[i] for i in range(len(y_std))]
        #         self.regression_error.iloc[:, k] +=err_weighted
        #     return (model, along_geo_pred,y_std)
        else:
            raise Exception("Wrong regression model: select either OLS or GPR")

    # Given x_new is predicting the corresponding graph:
    def predict(self,x_new,std=False):
        if(not isinstance(x_new,pd.core.frame.DataFrame)):
            print("The new observation should be a pandas dataframe of real values")
        elif (self.model_type == 'OLS'):
            self.y_vec_pred=self.model.predict(X=x_new)
            self.y_net_pred=GraphSet()
            for i in range(self.y_vec_pred.shape[0]):
                self.y_net_pred.add(self.give_me_a_network(geo=pd.Series(data=self.y_vec_pred[i],index=self.variables_names),n_a=self.aX.node_attr,e_a=self.aX.edge_attr,s=float(x_new.loc[i])))
        elif(self.model_type=='OLS_Anova'):
            x_new_dummy= pd.get_dummies(data=x_new, drop_first=True)
            self.y_vec_pred = self.model.predict(X=x_new_dummy)
            self.y_net_pred = GraphSet()
            for i in range(self.y_vec_pred.shape[0]):
                self.y_net_pred.add(
                    self.give_me_a_network(geo=pd.Series(data=self.y_vec_pred[i], index=self.variables_names),
                                           n_a=self.aX.node_attr, e_a=self.aX.edge_attr, s=int(x_new.loc[i])))

        # if(std==True and self.model_type=='GPR'):
        #     self.y_vec_pred, self.y_std_pred = self.model.predict(X=x_new,return_std=True)
        #     self.y_net_pred = GraphSet()
        #     for i in range(self.y_vec_pred.shape[0]):
        #         self.y_net_pred.add(
        #             self.give_me_a_network(geo=pd.Series(data=self.y_vec_pred[i], index=self.variables_names),
        #                                    n_a=self.aX.node_attr, e_a=self.aX.edge_attr, s=float(x_new.loc[i])))

    # These functions are auxiliary function to compute the ggr
    
    # geo is a pd Series
    # n_a and e_a are nodes and edges attributes
    def give_me_a_network(self,geo,n_a,e_a,s=None):
        ind=[re.findall(r'-?\d+\.?\d*', k) for k in geo.axes[0]]
        x_g={}
        for i in range(len(ind)):
            if(len(ind[i])>2 and int(ind[i][0])==int(ind[i][1]) and not (int(ind[i][0]),int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]),int(ind[i][1])]=[geo.loc[geo.axes[0][i+j]] for j in range(n_a)]
            elif(len(ind[i])>2 and int(ind[i][0])!=int(ind[i][1]) and not (int(ind[i][0]),int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]),int(ind[i][1])]=[geo.loc[geo.axes[0][i+j]] for j in range(e_a)]
            elif(len(ind[i])==2 and not (int(ind[i][0]),int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]),int(ind[i][1])]=[geo.loc[geo.axes[0][i]]]
        
        geo_net=Graph(x=x_g,adj=None,s=s)
        return geo_net



    # This function is used to parallelized the alignment procedure
    # receive two graphs, a matcher, an f where you are willing to save the permutations and gives back the
    # optimal permutation
    def two_net_match(self, X2, i,first_id):
        print("Matching Element" + str(i))
        if(i==first_id):
            self.f[first_id] = range(self.aX.n_nodes)
            print("Aligned" + str(i))
        # Align X to Y
        else:
            self.matcher.dis(self.aX.X[i], X2)
            # Permutation of X to go closer to Y
            self.f[i] = self.matcher.f
            print("Aligned"+str(i))