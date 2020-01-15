# The following class run the following algorithm approach:
# Take: X1,..,Xn obstervation in our Q quotient space
# Step1: Initialize the principal components
# Step2: Align X2,...,Xn wrt m (save the permutation vector)
# Step3: Compute the Principal Components
# Step4: m= Principal Components at current loop
# repreat step2-4 until m is not changing

from core import Graph
from core import GraphSet
from matcher import Matcher, BK, alignment, GA, ID
from distance import euclidean
from AlignCompute import aligncompute
import numpy as np
import itertools
import copy
import math
# from scipy.sparse.linalg import eigs
# from scipy.sparse import *
from sklearn.decomposition import NMF
from sklearn.preprocessing import scale
from sklearn import metrics
import re
import pandas as pd


class gnmf_vector(aligncompute):

    def __init__(self, graphset, matcher, distance):
        aligncompute.__init__(self, graphset, matcher)
        self.mean = None
        self.measure = distance

    def align_and_est(self, n_comp):
        # k=100 maximum number of iteration
        for k in range(100):
            # STEP 0: Align wrt mean, Compute the first nmf
            if (k == 0):
                self.f[0] = list(range(self.aX.n_nodes))
                m_1 = self.aX.X[0]
                for i in range(1, self.aX.size()):
                    # Align X to Y
                    a = self.matcher.align(self.aX.X[i], m_1)
                    # Permutation of X to go closer to Y
                    self.f[i] = a.f
                    # Compute the first Principal Component in the first step
                E_1 = self.est(n_comp)
                continue
                # return E1

            # STEP 1: Align wrt the first principal conponent
            self.align_geo(E_1[1].loc[0, :])
            # STEP 2: Compute the principal component
            if (k > 0):
                E_2 = self.est(n_comp)
            # STEP 3: Step range is the difference between the eigenvalues
            step_range = (E_2[0] - E_1[0])

            if (step_range < 0.01):
                # IF small enough, I am converging! Save and exit.
                self.e_val = E_2[0]
                self.scores = E_2[2]
                if (n_comp == 1):
                    self.e_vec = self.give_me_a_network(E_2[1].loc[0, :], n_a=self.aX.node_attr, e_a=self.aX.edge_attr)
                else:
                    G_NMF = GraphSet()
                    for n_nmf in range(n_comp):
                        G_NMF.add(
                            self.give_me_a_network(E_2[1].loc[n_nmf, :], n_a=self.aX.node_attr, e_a=self.aX.edge_attr))
                    self.e_vec = G_NMF
                print("Step Range smaller than 0.001")
                return
            else:
                # Go on with the computation: update the new result and restart from step 1.
                del E_1
                E_1 = E_2
                del E_2
        print("Maximum number of iteration reached.")
        # Return the result
        if ('E_2' in locals()):
            self.e_val = E_2[0]
            self.scores = E_2[2]
            if (n_comp == 1):
                self.e_vec = self.give_me_a_network(E_2[1].loc[0, :], n_a=self.aX.node_attr, e_a=self.aX.edge_attr)
            else:
                G_NMF = GraphSet()
                for n_nmf in range(n_comp):
                    G_NMF.add(
                        self.give_me_a_network(E_2[1].loc[n_nmf, :], n_a=self.aX.node_attr, e_a=self.aX.edge_attr))
                self.e_vec = G_NMF
                del G_NMF
            del E_2
        else:
            self.e_val = E_1[0]
            self.scores = E_1[2]
            if (n_comp == 1):
                self.e_vec = self.give_me_a_network(E_1[1].loc[0, :], n_a=self.aX.node_attr, e_a=self.aX.edge_attr)
            else:
                G_NMF = GraphSet()
                for n_nmf in range(n_comp):
                    G_NMF.add(
                        self.give_me_a_network(E_1[1].loc[n_nmf, :], n_a=self.aX.node_attr, e_a=self.aX.edge_attr))
                self.e_vec = G_NMF
                del G_NMF
            del E_1

    # Align wrt a geodesic
    def align_geo(self, geo):
        self.f.clear()
        # the alignment wrt a geodesic gamma(t) work in two step:
        # In this application, the geodesic gamma is a network so I need
        # to transform the vector into a network
        self.aX.get_node_attr()
        self.aX.get_edge_attr()

        geo_net = self.give_me_a_network(geo, n_a=self.aX.node_attr, e_a=self.aX.edge_attr)
        # step 1: every graph for every tilde_t in -T,T
        # Save the alignment for every i for every t_tilde in a dictionary
        for i in range(self.aX.size()):
            ind = 0
            f_i_t = {}
            d_i_t = []
            for tilde_t in range(-100, 100, 10):
                a = self.matcher.align(self.aX.X[i], geo_net.scale(tilde_t))
                d_i_t += [a.dis()]
                f_i_t[ind] = a.f
                ind += 1
                del a
            # step 2: find the best t_tilde for every i that minimize the distance
            t = np.argmin(d_i_t)
            self.f[i] = f_i_t[t]
            del ind, d_i_t, f_i_t

    # Est is computing the Covariance Matrix. The covariance matrix is the best choice
    # because it let you deal with different type of distance on node and edges
    def est(self, n_nmf):
        # dimension of the dataset
        N = self.aX.size()
        # Step 1: Create the current permuted dataset
        G_per = copy.deepcopy(self.aX)
        for i in range(N):
            G_per.X[i].permute(self.f[i])
        Mat = G_per.to_matrix_with_attr()
        # Standardizing the features
        #Mat_scale = pd.DataFrame(scale(Mat), columns=Mat.columns)
        nmf = NMF(n_components=n_nmf)
        scores = nmf.fit_transform(Mat)
        vals = metrics.explained_variance_score(y_true=Mat,y_pred=nmf.inverse_transform(scores))
        vecs = pd.DataFrame(nmf.components_, columns=Mat.columns)

        # top=np.argmax(vals_k)
        # TO HERE
        # vals=(vals_k[top]/sum(vals_k)).real
        # vecs=vecs_k[:,[top]]
        del Mat, G_per
        return (vals, vecs, scores)

    # add function  is the one used for computing the mean
    def add(self, ax, A, ay, B, f):
        # Adjency Matrix: x, y
        y = B.x
        G = copy.deepcopy(A)
        G.permute(f)
        x = G.x

        # coefficients: ax, ay
        # Links
        adjX = G.adj
        adjY = B.adj
        nY = B.n_nodes
        new = {}
        fullset = set(x.keys()).union(set(y.keys()))

        for i in range(nY):
            if ((i, i) in x and (i, i) in y):
                new[i, i] = self.summ(ax, x[i, i], ay, y[i, i])
            elif ((i, i) in x and not (i, i) in y):
                new[i, i] = self.summ(ax, x[i, i], ay, None)
            elif ((not (i, i) in x) and (i, i) in y):
                new[i, i] = self.summ(ax, None, ay, y[i, i])

            # degree=self.X.degree(i)
            linked_nodes = []
            if (i in adjX and i in adjY):
                linked_nodes = set(adjX[i]).union(set(adjY[i]))
            else:
                if (i in adjX and not i in adjY):
                    linked_nodes = set(adjX[i])
                if (i in adjY and not i in adjX):
                    linked_nodes = set(adjY[i])

            # for j in range(degree):
            for j in linked_nodes:

                if ((not (i, j) in y) and (not (i, j) in x)):
                    continue
                elif ((i, j) in y and (i, j) in x):
                    new[i, j] = self.summ(ax, x[i, j], ay, y[i, j])
                elif (not (i, j) in y):
                    # new[fi,fj]=self.summ(ax,x[i,j0],ay,[0]*len(x[i,j0]))
                    new[i, j] = self.summ(ax, x[i, j], ay, None)
                    # if(not x.has_key((i,j0))):
                elif (not (i, j) in x):
                    # new[fi,fj]=self.summ(ax,[0]*len(x[i,j0]),ay,y[fi,fj])
                    new[i, j] = self.summ(ax, None, ay, y[i, j])
        newG = Graph(x=new, y=None, adj=None)
        return newG

    # Add at y a linear combination of x y=ax*y + ay*x
    def summ(self, ax, x, ay, y):  # ax,ay are scalar, x,y are vectors
        if (x is None and y is None):
            return None
        else:
            if (x is None):
                res = [i * ay for i in y]
                return res  # ATTENTION: scalar moltiplication of scalar ay with vector y
            else:
                n = len(x)
                if (y is None):
                    y = np.zeros(n)
                res = []
                for i in range(n):
                    res += [ax * x[i] + ay * y[i]]
                return res

    # component wise distance function: usefull to compute the covariance
    def dis_componentwise(self, A, B, f):
        # Adjency Matrix: x, y
        y = B.x
        G = copy.deepcopy(A)
        G.permute(f)
        x = G.x
        # coefficients: ax, ay
        # Links
        adjX = G.adj
        adjY = B.adj
        nX = A.n_nodes
        new = {}
        fullset = set(x.keys()).union(set(y.keys()))
        for i in range(nX):
            if ((i, i) in x and (i, i) in y):
                new[i, i] = [math.sqrt(self.measure.node_dis(x[i, i], y[i, i]))]
            elif ((i, i) in x and not (i, i) in y):
                new[i, i] = [math.sqrt(self.measure.node_dis(x[i, i], [0]))]
            elif ((not (i, i) in x) and (i, i) in y):
                new[i, i] = [math.sqrt(self.measure.node_dis(y[i, i], [0]))]

            linked_nodes = []
            if (i in adjX and i in adjY):
                linked_nodes = set(adjX[i]).union(set(adjY[i]))
            else:
                if (i in adjX and not i in adjY):
                    linked_nodes = set(adjX[i])
                if (i in adjY and not i in adjX):
                    linked_nodes = set(adjY[i])

            for j in linked_nodes:
                # Both edges don't exist in both networks (impossible)
                if ((not (i, j) in y) and (not (i, j) in x)):
                    continue
                # Both edges exist in both networks
                elif ((i, j) in y and (i, j) in x):
                    new[i, j] = [math.sqrt(self.measure.edge_dis(x[i, j], y[i, j]))]
                elif (not (i, j) in y):
                    new[i, j] = [math.sqrt(self.measure.edge_dis(x[i, j], [0]))]
                elif (not (i, j) in x):
                    new[i, j] = [math.sqrt(self.measure.edge_dis([0], y[i, j]))]
        newG = Graph(x=new, y=None, adj=None)
        return newG

    # geo is a pd Series
    # n_a and e_a are nodes and edges attributes
    def give_me_a_network(self, geo, n_a, e_a):
        ind = [re.findall(r'-?\d+\.?\d*', k) for k in geo.axes[0]]
        x_g = {}
        for i in range(len(ind)):
            if (len(ind[i]) > 2 and int(ind[i][0]) == int(ind[i][1]) and not (int(ind[i][0]), int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]), int(ind[i][1])] = [geo.loc[geo.axes[0][i + j]] for j in range(n_a)]
            elif (len(ind[i]) > 2 and int(ind[i][0]) != int(ind[i][1]) and not (int(ind[i][0]), int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]), int(ind[i][1])] = [geo.loc[geo.axes[0][i + j]] for j in range(e_a)]
            elif (len(ind[i]) == 2 and not (int(ind[i][0]), int(ind[i][1])) in x_g):
                x_g[int(ind[i][0]), int(ind[i][1])] = [geo.loc[geo.axes[0][i]]]

        geo_net = Graph(x=x_g, adj=None, y=None)
        return geo_net