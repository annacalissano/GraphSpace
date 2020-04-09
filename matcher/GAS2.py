import pandas as pd
from matcher import Matcher
from core import GraphSet
import docplex.mp.model as cpx
from sklearn.metrics.pairwise import pairwise_distances

# Docplex approach

# GAS is a child of matcher
# GAS algorithm is used to compute the match between two networks through the usage of
# docplex python package and the cplex solver
# Giving two input networks, the algorithm choose the best matching between nodes by
# solving the associated optimization problem, minimizing the sum of pairwise distances
# between both nodes and edges. The input of cplex is a pairwise distance matrix.

# GAS2 would like to improve the performance of GAS by making the optimization problem linear
# More variables and constraints are introduced.

class GAS2(Matcher):

    def __init__(self,X=None,Y=None,f=None):
        Matcher.__init__(self,X,Y,f)
        self.name="Gaaaaaaas! - Linear"
        
    
    # The match function: this function find the best match among the equivalent classes
    def match(self,X,Y):
        # Take the two graphs - they have already the same size
        self.X=X
        self.Y=Y
        
        nX=self.X.nodes()
        set_I = range(nX)
        
        # Create Graph set:
        GG = GraphSet()
        GG.add(self.X)
        GG.add(self.Y)
        gg_mat = GG.to_matrix_with_attr()
        # building up the matrix of pairwise distances:
        gas = pd.DataFrame(pairwise_distances(gg_mat.iloc[[0]].transpose(),
                                              gg_mat.iloc[[1]].transpose()),
                           columns = gg_mat.iloc[[1]].columns,
                           index = gg_mat.iloc[[0]].columns)
        del GG, gg_mat
        
               
        # optimization model:
        # initialize the model
        opt_model = cpx.Model(name="HP Model", ignore_names=True, checker='off')

        # list of binary variables: 1 if i match j and u match v, 0 otherwise
        # x_vars is n x n x n x n
        x_vars  = {(i,j,u,v): opt_model.binary_var(name="x_{0}_{1}_{2}_{3}".format(i,j,u,v))
                   for i in set_I for j in set_I for u in set_I for v in set_I}
        
        # constraints - imposing that there is a one to one correspondence between the nodes in the two networks
        opt_model.add_constraints( (opt_model.sum(x_vars[i,i,u,u] for i in set_I)== 1 for u in set_I),
                                  ("constraint_r{0}".format(u) for u in set_I) )

        opt_model.add_constraints( (opt_model.sum(x_vars[i,i,u,u] for u in set_I)== 1 for i in set_I), 
                                  ("constraint_c{0}".format(i) for i in set_I)  )

        # we want to have: x_iu = 1 and x_jv = 1 <==> x_ijuv=1
        # the constraint x_iu = 1 and x_jv = 1 ==> x_ijuv=1
        # is written as x_ijuv >= x_iu + x_jv -1
        opt_model.add_constraints( ( x_vars[i,j,u,v] - x_vars[i,i,u,u] - x_vars[j,j,v,v]>= -1 
                                    for i in set_I for u in set_I 
                                    for j in set_I if j != i for v in set_I if v != u), 
                                  ("constraint_e{0}_{1}_{2}_{3}".format(i,j,u,v) 
                                   for i in set_I for u in set_I 
                                   for j in set_I if j != i for v in set_I if v != u)  )
        # to have also the opposite implication, we force x_ijuv to be zero
        # by adding a constant in the objective function - which has to be minimized

        # objective function - sum the distance between nodes and the distance between edges
        # e.g. (i,i) is a node in X, (u,u) is a node in Y, (i,j) is an edge in X, (u,v) is an edge in Y.
        objective = opt_model.sum(x_vars[i,i,u,u] * gas.loc['({0}, {0})'.format(i), '({0}, {0})'.format(u)]
                                  for i in set_I 
                                  for u in set_I) + opt_model.sum(x_vars[i,j,u,v] * (gas.loc['({0}, {1})'.format(i,j), 
                                                                                             '({0}, {1})'.format(u,v)]+ 0.01)
                                                                  for i in set_I 
                                                                  for u in set_I
                                                                  for j in set_I if j != i
                                                                  for v in set_I if v != u)

        # Minimizing the distances as specified in the objective function
        opt_model.minimize(objective)
        # Finding the minimum
        opt_model.solve()

        # Save in f the permutation: <3
        ff = {k:v.solution_value for k, v in x_vars.items()}
        self.f = [u for (i,j,u,v), k in ff.items() if (k == 1 and i==j and u==v)] 
        
        del gas

        # <3
 
            

