import pandas as pd
from matcher import Matcher
import docplex.mp.model as cpx
from sklearn.metrics.pairwise import pairwise_distances

# Docplex brute-force approach - variant (NOT STABLE, DO NOT USE)

# GAS1 would like to improve the computations for the pairwise distance matrix
# by using only the relevant subset of (i,j) edges/nodes common to X and Y.
# In this way, the matrix is smaller and hopefully faster to deal with.

class GAS1(Matcher):

    def __init__(self,X=None,Y=None,f=None):
        Matcher.__init__(self,X,Y,f)
        self.name="Gaaaaaaas!"
        
    
    # The match function: this function find the best match among the equivalent classes
    def match(self,X,Y):
        # Take the two graphs - they have already the same size
        self.X=X
        self.Y=Y
        
        nX=self.X.nodes()
        # nY=self.Y.nodes()
        set_I = range(nX)
        
        # building up the matrix of pairwise distances:

        # GAS V0 - not working properly, if null edges are not defined (to 0.0)
#        gas = pd.DataFrame(pairwise_distances(X.to_vector_with_attributes().transpose(),
#                                              Y.to_vector_with_attributes().transpose()),
#                           columns = Y.to_vector_with_attributes().columns,
#                           index = X.to_vector_with_attributes().columns)

        
        # GAS V1 - slow
        # Create Graph set:
#        GG = GraphSet()
#        GG.add(self.X)
#        GG.add(self.Y)
#        gg_mat = GG.to_matrix_with_attr()
#        gas = pd.DataFrame(pairwise_distances(gg_mat.iloc[[0]].transpose(),
#                                              gg_mat.iloc[[1]].transpose()),
#                           columns = gg_mat.iloc[[1]].columns,
#                           index = gg_mat.iloc[[0]].columns)
#        del GG, gg_mat
        
        # GAS V2
        # set of non-zero indices (i,j) that are in X or in Y
        iset = set([x for (x,y) in self.X.x.items() if y!=[0]]).union(set([x for (x,y) in self.Y.x.items() if y!=[0]]))
        # # set of indices (i,j) that are in X or in Y
        # iset = set(self.X.x.keys()).union(set(self.Y.x.keys()))
        x_vec = X.to_vector_with_select_attributes(iset)
        y_vec = Y.to_vector_with_select_attributes(iset)
        gas = pd.DataFrame(pairwise_distances(x_vec.transpose(),
                                              y_vec.transpose()),
                           columns = y_vec.columns,
                           index = x_vec.columns)
        del x_vec, y_vec
        
        # GAS V3
#        GG.get_node_attr()
#        vip1 = set([k and v for ((k,v),j) in GG.X[1].x.items() if j!=[0]*GG.node_attr]) # and v
#        vip2 = set([k and v for ((k,v),j) in GG.X[2].x.items() if j!=[0]*GG.node_attr])
#        vip =vip1.union(vip2)

                
        # optimization model:
        # initialize the model
        opt_model = cpx.Model(name="HP Model")

        # list of binary variables: 1 if i match j, 0 otherwise
        # x_vars is n x n
        x_vars  = {(i,u): opt_model.binary_var(name="x_{0}_{1}".format(i,u))
                   for i in set_I for u in set_I}
        
        # constraints - imposing that there is a one to one correspondece between the nodes in the two networks
        # note. the constrains are created in opt_model.add_constraint
        constraint_sr = {u : opt_model.add_constraint(ct=opt_model.sum(x_vars[i,u] for i in set_I) 
                                                    == 1,ctname="constraint_{0}".format(u)) for u in set_I}

        constraints_cr = {(nX+i) : opt_model.add_constraint(ct=opt_model.sum(x_vars[i,u] for u in set_I) 
                                                    == 1,ctname="constraint_{0}".format(i)) for i in set_I} 
        
        # objective function - sum the distance between nodes and the distance between edges
        # e.g. (i,i) is a node in X, (u,u) is a node in Y, (i,j) is an edge in X, (u,v) is an edge in Y.

#        objective = opt_model.sum(x_vars[i,u] * gas.loc['({0}, {0})'.format(i), '({0}, {0})'.format(u)]
#                                  for i in set_I 
#                                  for u in set_I) + opt_model.sum(x_vars[i,u] * x_vars[j,v] * gas.loc['({0}, {1})'.format(i,j), 
#                                                                                                      '({0}, {1})'.format(u,v)]
#                                                                  for i in set_I 
#                                                                  for u in set_I
#                                                                  for j in set_I if j != i
#                                                                  for v in set_I if v != u)
        
        set_I = set([i for (i,i) in iset]) 
        objective = opt_model.sum(x_vars[i,u] * gas.loc['({0}, {0})'.format(i), '({0}, {0})'.format(u)]
                                  for i in set_I 
                                  for u in set_I) + opt_model.sum(x_vars[i,u] * x_vars[j,v] * gas.loc['({0}, {1})'.format(i,j), 
                                                                                                      '({0}, {1})'.format(u,v)]
                                                                  for (i,j) in iset if i!=j
                                                                  for (u,v) in iset if u!=v)
        # Minimizing the distances as specified in the objective function
        opt_model.minimize(objective)
        # Finding the minimum
        opt_model.solve()

        # Save in f the permutation: <3
        ff = {k:v.solution_value for k, v in x_vars.items()}
        self.f = [k for (j,k), v in ff.items() if v == 1]
        
        del gas

        # <3
 
            

