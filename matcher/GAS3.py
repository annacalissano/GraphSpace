import pandas as pd
from matcher import Matcher
from core import GraphSet
import docplex.mp.model as cpx
from sklearn.metrics.pairwise import pairwise_distances

# Docplex approach

# GAS3() is used to try and test possible variants of the code, to see if there is a corresponding
# improvement in performance.
# In particular, in this version we try to access the gas matrix values in a different way - with no success.

class GAS3(Matcher):

    def __init__(self,X=None,Y=None,f=None):
        Matcher.__init__(self,X,Y,f)
        self.name="Gaaaaaaas!"

    # The match function: this function find the best match among the equivalent classes
    def match(self, X, Y):
        # Take the two graphs - they have already the same size
        self.X = X
        self.Y = Y

        nX = self.X.nodes()
        set_I = range(nX)

        # building up the matrix of pairwise distances:

        # GAS V1
        # Create Graph set:
        GG = GraphSet()
        GG.add(self.X)
        GG.add(self.Y)
        gg_mat = GG.to_matrix_with_attr()
        gas = pd.DataFrame(pairwise_distances(gg_mat.iloc[[0]].transpose(),
                                              gg_mat.iloc[[1]].transpose()),
                           columns=gg_mat.iloc[[1]].columns,
                           index=gg_mat.iloc[[0]].columns)
        del GG, gg_mat

        # optimization model:
        # initialize the model
        opt_model = cpx.Model(name="HP Model", ignore_names=True, checker='off')

        # list of binary variables: 1 if i match j, 0 otherwise
        # x_vars is n x n x n x n
        x_vars = {(i, j, u, v): opt_model.binary_var(name="x_{0}_{1}_{2}_{3}".format(i, j, u, v))
                  for i in set_I for j in set_I for u in set_I for v in set_I}

        # constraints - imposing that there is a one to one correspondence between the nodes in the two networks
        opt_model.add_constraints_((opt_model.sum(x_vars[i, i, u, u] for i in set_I) == 1 for u in set_I),
                                   ("constraint_r{0}".format(u) for u in set_I))

        opt_model.add_constraints_((opt_model.sum(x_vars[i, i, u, u] for u in set_I) == 1 for i in set_I),
                                   ("constraint_c{0}".format(i) for i in set_I))

        opt_model.add_constraints_((x_vars[i, j, u, v] - x_vars[i, i, u, u] - x_vars[j, j, v, v] >= -1
                                    for i in set_I for u in set_I
                                    for j in set_I if j != i for v in set_I if v != u),
                                   ("constraint_e{0}_{1}_{2}_{3}".format(i, j, u, v)
                                    for i in set_I for u in set_I
                                    for j in set_I if j != i for v in set_I if v != u))

        # objective function - sum the distance between nodes and the distance between edges
        # e.g. (i,i) is a node in X, (u,u) is a node in Y, (i,j) is an edge in X, (u,v) is an edge in Y.

        objective = opt_model.sum(x_vars[i, i, u, u] * gas.iloc[(nX+1)*i, (nX+1)*u]
                                  for i in set_I
                                  for u in set_I) + opt_model.sum(
            x_vars[i, j, u, v] * (gas.iloc[nX*i + j, nX*u + v] + 0.01)
            for i in set_I
            for u in set_I
            for j in set_I if j != i
            for v in set_I if v != u)

        # Minimizing the distances as specified in the objective function
        opt_model.minimize(objective)
        # Finding the minimum
        opt_model.solve()

        # Save in f the permutation: <3
        ff = {k: v.solution_value for k, v in x_vars.items()}
        self.f = [u for (i, j, u, v), k in ff.items() if (k == 1 and i == j and u == v)]

        del gas

        # <3
            

