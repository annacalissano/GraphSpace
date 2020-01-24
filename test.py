from core import Graph, GraphSet, Mean, MeanIterative
from distance import euclidean
from matcher import GA, ID
from AlignCompute import mean_aac, gpc_aac

from core import Graph
from core import GraphSet
from core import Mean
from core import MeanIterative
from matcher import Matcher, alignment, GA, ID
from distance import euclidean

import math
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, vstack

# Define the graphs:
x1 = {}
x1[0, 0] = [0.813, 0.630]
x1[1, 1] = [1.606, 2.488]
x1[2, 2] = [2.300, 0.710]
x1[3, 3] = [0.950, 1.616]
x1[4, 4] = [2.046, 1.560]
x1[5, 5] = [2.959, 2.387]
x1[0, 1] = [1]
x1[1, 0] = [1]
x1[1, 2] = [1]
x1[2, 1] = [1]
x1[2, 5] = [1]
x1[3, 4] = [1]
x1[4, 3] = [1]
x1[5, 2] = [1]
x2 = {}
x2[0, 0] = [0.810, 0.701]
x2[1, 1] = [1.440, 2.437]
x2[2, 2] = [2.358, 0.645]
x2[3, 3] = [0.786, 1.535]
x2[4, 4] = [2.093, 1.591]
x2[5, 5] = [3.3, 2.2]
x2[0, 1] = [1]
x2[1, 0] = [1]
x2[1, 2] = [1]
x2[2, 1] = [1]
x2[3, 4] = [1]
x2[4, 3] = [1]
x3 = {}
x3[0, 0] = [0.71, 0.72]
x3[1, 1] = [1.45532, 2.45648]
x3[2, 2] = [2.21121, 0.757368]
x3[3, 3] = [0.796224, 1.53137]
x3[4, 4] = [2.06496, 1.5699]
x3[5, 5] = [2.75535, 0.194153]
x3[0, 1] = [1]
x3[1, 0] = [1]
x3[0, 5] = [1]
x3[5, 0] = [1]
x3[1, 2] = [1]
x3[2, 1] = [1]
x3[3, 4] = [1]
x3[4, 3] = [1]

# Create Graph set:
G = GraphSet()
G.add(Graph(x=x1, y=None, adj=None))
G.add(Graph(x=x2, y=None, adj=None))
G.add(Graph(x=x3, y=None, adj=None))

# or import a GraphSet
X = GraphSet()
X.read_from_text("Simulation_Data.txt")

# Matching Function - GA or ID:
match = GA()
match.dis(G.X[0], G.X[1])
# to see the matching transformation:
print(match.f)
del match

# Compute the mean with GA matcher
match = ID()
mu = Mean(G, match)
MU = mu.mean()
# to see the result:
print(MU.x)

del match, mu, MU

# Align All and Compute Mean
match = GA()
mu = mean_aac(G, match)
mu.align_and_est()
MU = mu.mean
print(MU.x)

# Align All and Compute GPC
n_comp=2
p=gpc_aac(G,GA(),euclidean())
p.align_and_est(n_comp,scale=False,s=[0,10])
