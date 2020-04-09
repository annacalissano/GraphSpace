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

        # set of non-zero nodes (i,i) that are in X or in Y
        # note. assuming that if there is an edge (i,j), both i and j have non-zero attribute
        isetn = set((i, j) for ((i, j), y) in self.X.x.items() if y != [0] if i == j).union(
            set((i, j) for ((i, j), y) in self.Y.x.items() if y != [0] if i == j))
        # set of indices i of non-zero nodes (i,i) that are in X or in Y
        isetnn = {i for (i, j) in isetn}
        # set of edges btw non-zero nodes that are in X or in Y
        isete = {(i, j) for i in isetnn for j in isetnn if i != j}

        # building up the matrix of pairwise distances btw nodes:
        x_vec_n = self.X.to_vector_with_select_attributes(isetn)
        y_vec_n = self.Y.to_vector_with_select_attributes(isetn)
        gas_n = pd.DataFrame(pairwise_distances(x_vec_n.transpose(),
                                                y_vec_n.transpose()),
                             columns=y_vec_n.columns,
                             index=x_vec_n.columns)
        del x_vec_n, y_vec_n

        # building up the matrix of pairwise distances btw edges:
        x_vec_e = self.X.to_vector_with_select_attributes(isete)
        y_vec_e = self.Y.to_vector_with_select_attributes(isete)
        gas_e = pd.DataFrame(pairwise_distances(x_vec_e.transpose(),
                                                y_vec_e.transpose()),
                             columns=y_vec_e.columns,
                             index=x_vec_e.columns)
        del x_vec_e, y_vec_e
        
        # optimization model:

        # initialize the model
        opt_model = cpx.Model(name="HP Model")

        # list of binary variables: 1 if i match j, 0 otherwise
        # x_vars is n x n
        x_vars = {(i, u): opt_model.binary_var(name="x_{0}_{1}".format(i, u))
                  for i in isetnn
                  for u in isetnn}

        # constraints - imposing that there is a one to one correspondence between the nodes in the two networks
        opt_model.add_constraints_((opt_model.sum(x_vars[i, u] for i in isetnn) == 1
                                    for u in isetnn),
                                   ("constraint_r{0}".format(u) for u in isetnn))

        opt_model.add_constraints_((opt_model.sum(x_vars[i, u] for u in isetnn) == 1
                                    for i in isetnn),
                                   ("constraint_c{0}".format(i) for i in isetnn))

        # opt_model.add_constraints_((opt_model.sum(x_vars[i,u] for i in {i for (i,w) in iset if w ==u}) == 1
        #                             for u in isety),
        #                            ("constraint_r{0}".format(u) for u in isety))
        # constraint_sr = {u : (opt_model.sum(x_vars[i,u] for i in {i for (i,u) in iset})) for u in isety}
        # constraint_sr = {u2: (opt_model.sum(x_vars[i, u2] for i in {i for (i, u2) in iset})) for u2 in isety}
        #
        # constraint_sr = {u : opt_model.add_constraint(ct=opt_model.sum(x_vars[i,u] for (i,u) in iset)
        #                                             == 1,ctname="constraint_{0}".format(u)) for u in set_I}
        #
        # constraints_cr = {(nX+i) : opt_model.add_constraint(ct=opt_model.sum(x_vars[i,u] for u in set_I)
        #                                             == 1,ctname="constraint_{0}".format(i)) for i in set_I}
        
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

        objective = opt_model.sum(x_vars[i, u] * gas_n.loc['({0}, {0})'.format(i), '({0}, {0})'.format(u)]
                                  for i in isetnn
                                  for u in isetnn) + opt_model.sum(
            x_vars[i, u] * x_vars[j, v] * gas_e.loc['({0}, {1})'.format(i, j),
                                                    '({0}, {1})'.format(u, v)]
            for (i, j) in isete
            for (u, v) in isete)

        # Minimizing the distances as specified in the objective function
        opt_model.minimize(objective)
        # Finding the minimum
        opt_model.solve()

        # Save in f the permutation: <3
        ff = [k for k, v in x_vars.items() if v.solution_value == 1]
        if len(ff) < nX:
            # if the number of nodes involved in the matching, i.e. non-zero nodes, is smaller than the total,
            # set up the permutation vector in the proper way
            # e.g. X nodes 1,3 Y nodes 1,4 -> isetnn={1,3,4} -> len(x_vars>0)=3
            # -> i want to avoid st. like f=[4,1,3] because i want len(f)=nX
            self.f = list(range(nX))
            for (i, k) in ff:
                self.f[i] = k
        else:
            self.f = [k for (j, k) in ff]

        del gas_n, gas_e

        # <3
 
            

