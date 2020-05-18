import pandas as pd
from matcher import Matcher
import docplex.mp.model as cpx
from sklearn.metrics.pairwise import pairwise_distances, _VALID_METRICS
import copy

# Docplex approach

# GAS is a child of matcher
# GAS algorithm is used to compute the match between two networks through the usage of
# docplex python package and the cplex solver
# Giving two input networks, the algorithm choose the best matching between nodes by
# solving the associated optimization problem, minimizing the sum of pairwise distances
# between both nodes and edges. The input of cplex is a pairwise distance matrix.

# NOTE: valid metrics are ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis',
# 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
# 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
# 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'haversine']

class GAS(Matcher):

    def __init__(self, X=None, Y=None, f=None, measure=None):
        Matcher.__init__(self, X, Y, f, measure)
        # measure can be a string - for both nodes and edges attributes
        if isinstance(self.measure, str):
            self.metricNode = self.metricEdge = self.measure
        # or a list of two strings
        elif isinstance(self.measure, list):
            self.metricNode = self.measure[0]
            self.metricEdge = self.measure[1]
        # or a measure defined in a proper way in the distance folder
        else:
            # if sklearn has already it implemented
            if self.measure.get_Instance() in _VALID_METRICS:
                self.metricNode = self.metricEdge = self.measure.get_Instance()
            # if the measure is really user-defined
            else:
                self.metricNode = self.measure.node_dis
                self.metricEdge = self.measure.edge_dis
        self.name = "Gaaaaaaas!"

    # The match function: this function find the best match among the equivalent classes
    # storeDistance is a boolean: if True, the value of the minimized objective function
    #                             is stored (in self.distance)
    def match(self, X, Y, storeDistance=False):
        # Take the two graphs - they have already the same size
        self.X = X
        self.Y = Y

        nX = self.X.nodes()

        # set of non-zero nodes (i,i) that are in X or in Y
        # note. assuming that if there is an edge (i,j), both i and j have non-zero attribute
        isetn = set((i, j) for ((i, j), y) in self.X.x.items() if y != [0] if i == j).union(
            set((i, j) for ((i, j), y) in self.Y.x.items() if y != [0] if i == j))
        isetn = sorted(isetn)
        # set of indices i of non-zero nodes (i,i) that are in X or in Y
        isetnn = [i for (i, j) in isetn]
        # set of edges btw non-zero nodes that are in X or in Y
        isete = [(i, j) for i in isetnn for j in isetnn if i != j]

        # building up the matrix of pairwise distances:
        #   matrix of pairwise distances btw nodes:
        x_vec_n = self.X.to_vector_with_select_nodes(isetn)
        y_vec_n = self.Y.to_vector_with_select_nodes(isetn)
        gas_n = pd.DataFrame(pairwise_distances(x_vec_n,
                                                y_vec_n,
                                                metric=self.metricNode),
                             columns=y_vec_n.index,
                             index=x_vec_n.index)
        del x_vec_n, y_vec_n

        #   matrix of pairwise distances btw edges:
        try:
            x_vec_e = self.X.to_vector_with_select_edges(isete)
            y_vec_e = self.Y.to_vector_with_select_edges(isete)
            gas_e = pd.DataFrame(pairwise_distances(x_vec_e,
                                                    y_vec_e,
                                                    metric=self.metricEdge),
                                 columns=y_vec_e.index,
                                 index=x_vec_e.index)
            del x_vec_e, y_vec_e
        except:
            # degenerate graphs
            if self.X.edge_attr + self.Y.edge_attr == 0:  # if both the two graphs have no edge
                gas_e = pd.DataFrame()
            if self.X.edge_attr * self.Y.edge_attr == 0:  # if one of the two graphs has no edge
                self.X.edge_attr = self.Y.edge_attr = max(self.X.edge_attr, self.Y.edge_attr)
                x_vec_e = self.X.to_vector_with_select_edges(isete)
                y_vec_e = self.Y.to_vector_with_select_edges(isete)
                gas_e = pd.DataFrame(pairwise_distances(x_vec_e,
                                                        y_vec_e,
                                                        metric=self.metricEdge),
                                     columns=y_vec_e.index,
                                     index=x_vec_e.index)
                del x_vec_e, y_vec_e

        # optimization model:
        # initialize the model
        # opt_model = cpx.Model(name="HP Model")
        opt_model = cpx.Model(name="HP Model", ignore_names=True, checker='off')  # faster

        # list of binary variables: 1 if i match j, 0 otherwise
        # x_vars is n x n
        # x_vars = {(i, u): opt_model.binary_var(name="x_{0}_{1}".format(i, u))
        #           for i in isetnn
        #           for u in isetnn}
        x_vars = opt_model.binary_var_matrix(isetnn, isetnn, name="x")

        # constraints - imposing that there is a one to one correspondence between the nodes in the two networks
        opt_model.add_constraints_((opt_model.sum(x_vars[i, u] for i in isetnn) == 1
                                    for u in isetnn),
                                   (f"constraint_r{u}" for u in isetnn))

        opt_model.add_constraints_((opt_model.sum(x_vars[i, u] for u in isetnn) == 1
                                    for i in isetnn),
                                   (f"constraint_c{i}" for i in isetnn))

        # objective function - sum the distance between nodes and the distance between edges
        # e.g. (i,i) is a node in X, (u,u) is a node in Y, (i,j) is an edge in X, (u,v) is an edge in Y.
        objective = opt_model.sum(x_vars[i, u] * gas_n.loc[f'({i}, {i})', f'({u}, {u})']
                                  for i in isetnn
                                  for u in isetnn) + opt_model.sum(
            x_vars[i, u] * x_vars[j, v] * gas_e.loc[f'({i}, {j})',
                                                    f'({u}, {v})']
            for (i, j) in isete   # for i in isetnn for j in isetnn if j!=i
            for (u, v) in isete)  # for u in isetnn for v in isetnn if v!=u

        # Minimizing the distances as specified in the objective function
        opt_model.minimize(objective)
        # Finding the minimum
        opt_model.solve()
        if storeDistance:
            self.distance = opt_model.solution.get_objective_value()

        # Save in f the permutation: <3
        ff = [k for k, z in x_vars.items() if z.solution_value == 1]
        if len(ff) < nX:
            # if the number of nodes involved in the matching, i.e. non-zero nodes, is smaller than the total,
            # set up the permutation vector in the proper way
            # e.g. X nodes 1,3 Y nodes 1,4 -> isetnn={1,3,4} -> len(x_vars>0)=3
            # -> i want to avoid st. like f=[4,1,3] because i want len(f)=nX
            self.f = list(range(nX))
            for (i, u) in ff:
                self.f[i] = u
        else:
            self.f = [u for (i, u) in ff]

        del gas_n, gas_e

        # <3

    # Computing distance between two graph
    # NOTE: overwrite the matcher method, since computing the match gives us directly the result
    def the_dis(self, X, Y):
        # match gives back the best combination of nodes
        self.the_grow_and_set(X, Y)
        aX = copy.deepcopy(self.X)
        aY = copy.deepcopy(self.Y)
        self.match(aX, aY, storeDistance=True)
        return self.distance

    # Computing similarity between two graph
    # NOTE: if, in the instantiation, measure is a string, the_sim does not work
    def the_sim(self, X, Y):
        assert (not isinstance(self.measure, str) or not isinstance(self.measure, list)), "not implemented: \
        change distance \n distance must be a distance object with node_sim and edge_sim methods"
        return Matcher.the_sim(self, X, Y)
