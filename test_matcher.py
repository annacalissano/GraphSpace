# This is a test_matcher.py code to test the performance of the selected matcher
# In particular, it compares GA, GAS and GAS1 (but it works for any other matcher)
# for different graphsets

# Import packages
from AlignCompute import mean_aac
from core import Graph
from core import GraphSet
from matcher import Matcher, alignment, GA, ID, GAS, GAS1
from distance import euclidean

import numpy as np
from scipy import stats
from scipy.sparse import lil_matrix, vstack
import time

# np.random.seed(10)

####################################################
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
X.read_from_text("C:\\Users\\Gianluca\\Documents\\i-UNI\\Research\\Network\\CODE\\GraphSpace\\Dataset.txt")

# import the Eptagones GraphSet
X2 = GraphSet()
X2.read_from_text(
    "C://Users//Gianluca//Documents//i-UNI//Research//Network//CODE//GraphSpace/Pentagones_10_100_500_Perm.txt")

# or use a simpler 2-nodes Graphset
Y = GraphSet()
for j in range(45):
    x0 = {}
    i0 = np.random.binomial(n=1, p=0.4)
    x0[0, 0] = [1]
    x0[1, 1] = [1]
    x0[i0, (1 - i0)] = [np.random.normal(loc=20, scale=3)]
    Y.add(Graph(x=x0, y=None, adj=None))
for j in range(5):
    x0 = {}
    x0[0, 0] = [1]
    x0[1, 1] = [1]
    x0[0, 1] = [np.random.normal(loc=20, scale=3)]
    x0[1, 0] = [np.random.normal(loc=5, scale=1)]
    Y.add(Graph(x=x0, y=None, adj=None))

# or a simple 3-nodes Graphset
YY = GraphSet()
for j in range(5):
    x0 = {}
    i0 = np.random.binomial(n=1, p=0.4)
    x0[0, 0] = [1]
    x0[1, 1] = [1]
    x0[2, 2] = [1]
    x0[i0, (1 - i0)] = [np.random.normal(loc=20, scale=3)]
#    x0[(1 - i0), i0] = [0]
    x0[(i0 + 1), (2 - i0)] = [np.random.normal(loc=20, scale=3)]
    x0[(2 - i0), (i0 + 1)] = [np.random.normal(loc=5, scale=1)]
    x0[(i0 * 2), (2 - 2 * i0)] = [np.random.normal(loc=20, scale=3)]
    x0[(2 - 2 * i0), (2 * i0)] = [np.random.normal(loc=5, scale=1)]
    YY.add(Graph(x=x0, y=None, adj=None))

# or use a fixed-value simple 2-nodes Graphset
YZ = GraphSet()
x0 = {}
x0[0, 0] = [1]
x0[1, 1] = [8]
x0[1, 0] = [30]
x0[0, 1] = [10]
YZ.add(Graph(x=x0, y=None, adj=None))
x0 = {}
x0[0, 0] = [7]
x0[1, 1] = [3]
x0[0, 1] = [25]
x0[1, 0] = [15]
YZ.add(Graph(x=x0, y=None, adj=None))

# Y.to_matrix_with_attr().loc[0:5,:]

# YY.to_matrix_with_attr()


####################################################
############# First Graphset: Y ####################
#---------------------------------------------------


# Align All and Compute Mean
start = time.time()
match_ga = GA()
mu_ga = mean_aac(Y, match_ga)
mu_ga.align_and_est()
elapsed_time_ga = (time.time() - start)
print(elapsed_time_ga)

# GAS
start = time.time()
match = GAS()
mu = mean_aac(Y, match)
mu.align_and_est()
elapsed_time_gas = (time.time() - start)
print(elapsed_time_gas)

# GAS1: GAS linear version
start = time.time()
match1 = GAS1()
mu1 = mean_aac(Y, match1)
mu1.align_and_est()
elapsed_time_gas1 = (time.time() - start)
print(elapsed_time_gas1)


### print solutions:
# GA
print(mu_ga.aX.to_matrix_with_attr().loc[0:5,:])
mu_ga.f.values()
MU_ga = mu_ga.mean
print(MU_ga.x)
# GAS
print(mu.aX.to_matrix_with_attr().loc[0:5,:])
mu.f.values()
MU = mu.mean
print(MU.x)
# GAS1
print(mu1.aX.to_matrix_with_attr().loc[0:5,:])
mu1.f.values()
MU1 = mu1.mean
print(MU1.x)


####################################################
############ Second Graphset: YY ###################
#---------------------------------------------------

# Sometimes YY is not even solved by mean_aac, but even when it happens, improvements in speed are quite obvious.

# Align All and Compute Mean
start = time.time()
match_ga = GA()
mu_ga = mean_aac(YY, match_ga)
mu_ga.align_and_est()
elapsed_time_ga = (time.time() - start)
print(elapsed_time_ga)

# GAS
start = time.time()
match = GAS()
mu = mean_aac(YY, match)
mu.align_and_est()
elapsed_time_gas = (time.time() - start)
print(elapsed_time_gas)

# GAS1: GAS linear version
start = time.time()
match1 = GAS1()
mu1 = mean_aac(YY, match1)
mu1.align_and_est()
elapsed_time_gas1 = (time.time() - start)
print(elapsed_time_gas1)


####################################################
######### Third Graphset: Eptagones ################
#---------------------------------------------------

# Align All and Compute Mean
start = time.time()
match_ga = GA()
mu_ga = mean_aac(X2, match_ga)
mu_ga.align_and_est()
elapsed_time_ga = (time.time() - start)
print(elapsed_time_ga)

# GAS
start = time.time()
match = GAS()
mu = mean_aac(X2, match)
mu.align_and_est()
elapsed_time_gas = (time.time() - start)
print(elapsed_time_gas)

# GAS1: GAS linear version
start = time.time()
match1 = GAS1()
mu1 = mean_aac(X2, match1)
mu1.align_and_est()
elapsed_time_gas1 = (time.time() - start)
print(elapsed_time_gas1)


####################################################
############# Fourth Graphset: G ###################
#---------------------------------------------------

# Align All and Compute Mean
start = time.time()
match_ga = GA()
mu_ga = mean_aac(G, match_ga)
mu_ga.align_and_est()
elapsed_time_ga = (time.time() - start)
print(elapsed_time_ga)
mu_ga.f.values()

# GAS
start = time.time()
match = GAS()
mu = mean_aac(G, match)
mu.align_and_est()
elapsed_time_gas = (time.time() - start)
print(elapsed_time_gas)

# GAS1: GAS linear version
start = time.time()
match1 = GAS1()
mu1 = mean_aac(G, match1)
mu1.align_and_est()
elapsed_time_gas1 = (time.time() - start)
print(elapsed_time_gas1)
print(mu1.f.values())

### print solutions:
# GA
print(mu_ga.aX.to_matrix_with_attr().loc[0:5,:])
mu_ga.f.values()
MU_ga = mu_ga.mean
print(MU_ga.x)
# GAS
print(mu.aX.to_matrix_with_attr().loc[0:5,:])
mu.f.values()
MU = mu.mean
print(MU.x)
# GAS1
print(mu1.aX.to_matrix_with_attr().loc[0:5,:])
mu1.f.values()
MU1 = mu1.mean
print(MU1.x)


#print(MU.x)
#print(match.the_dis(MU, G.X[0]) + match.the_dis(MU, G.X[1]) + match.the_dis(MU, G.X[2]))
#print(match.the_dis(MU, G.X[0])**2 + match.the_dis(MU, G.X[1])**2 + match.the_dis(MU, G.X[2])**2)
