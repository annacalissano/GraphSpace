from core import Graph
from core import GraphSet
import gpcc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

from AlignCompute import gpc_aac
from matcher import BK, GA, ID
import time
import pandas
import os
import sys


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm

X = GraphSet()
X.read_from_text("Pentagones_10_100_500_Perm.txt")
#X.read_from_text("Mammals_Grooming_kickouts.txt")

match1=GA()
match2=ID()

gpc_c=gpc_aac(X, match1, cg=True)
gpc_p=gpc_aac(X,match1)

start1=time.time()
gpc_c.align_and_est(3,False,[0,10])
end1=time.time()
print("Tempo Impiegato:"+str(end1-start1))

start2=time.time()
gpc_p.align_and_est(3,False,[0,10])
end2=time.time()
print("Tempo Impiegato:"+str(end2-start2))


pos={0: [-1.,  -2.],
1: [-2., 0.        ],
4: [ 0., 2.],
3: [2. , 0.],
2: [1. , -2.] }

#PLOT WITH C++ BINDING

n_gpc=0
Vector=gpc_c.e_vec.X[n_gpc]
Bar=gpc_c.barycenter_net
l=list(np.sort(gpc_c.scores[:,n_gpc]))

percentiles=list()
percentiles.append(np.percentile(l, 1))
percentiles.append(np.percentile(l, 25))
percentiles.append(np.percentile(l, 50))
percentiles.append(np.percentile(l, 75))
percentiles.append(np.percentile(l, 90))

G_along_GPC1=GraphSet()
for i in range(len(percentiles)):
    G_along_GPC1.add(gpc_c.add(1,Bar,percentiles[i],Vector,range(Vector.n_nodes)))

plt.figure()
norm = mpl.colors.Normalize(vmin=-120, vmax=120, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='seismic')


for i in range(len(G_along_GPC1.X)):
    G_plot1=G_along_GPC1.X[i].to_networkX(layer=0, node_too=True)
    #pos=nx.kamada_kawai_layout(G_plot1)
    colors=list(nx.get_edge_attributes(G_plot1,'weight').values())
    colors_nodes=list(nx.get_node_attributes(G_plot1,'weight').values())
    to_rgb_col=[mapper.to_rgba(i) for i in colors]
    to_rgb_col_nodes=[mapper.to_rgba(i) for i in colors_nodes]
    # Draw the network
    nx.draw(G_plot1, pos, node_color=to_rgb_col_nodes, edge_color=to_rgb_col, width=4,#,norm=norm,
           with_labels=False)
    plt.show()

n_gpc=1
Vector=gpc_c.e_vec.X[n_gpc]
Bar=gpc_c.barycenter_net
l=list(np.sort(gpc_c.scores[:,n_gpc]))

percentiles=list()
percentiles.append(np.percentile(l, 1))
percentiles.append(np.percentile(l, 25))
percentiles.append(np.percentile(l, 50))
percentiles.append(np.percentile(l, 75))
percentiles.append(np.percentile(l, 90))

G_along_GPC3=GraphSet()
for i in range(len(percentiles)):
    G_along_GPC3.add(gpc_c.add(1,Bar,percentiles[i],Vector,range(Vector.n_nodes)))

plt.figure()
norm = mpl.colors.Normalize(vmin=-120, vmax=120, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='seismic')


for i in range(len(G_along_GPC.X)):
    G_plot1=G_along_GPC3.X[i].to_networkX(layer=0, node_too=True)
    #pos=nx.kamada_kawai_layout(G_plot1)
    colors=list(nx.get_edge_attributes(G_plot1,'weight').values())
    colors_nodes=list(nx.get_node_attributes(G_plot1,'weight').values())
    to_rgb_col=[mapper.to_rgba(i) for i in colors]
    to_rgb_col_nodes=[mapper.to_rgba(i) for i in colors_nodes]
    # Draw the network
    nx.draw(G_plot1, pos, node_color=to_rgb_col_nodes, edge_color=to_rgb_col, width=4,#,norm=norm,
           with_labels=False)
    plt.show()


#PLOT FOR PYTHON

n_gpc=0
Vector=gpc_p.e_vec.X[n_gpc]
Bar=gpc_p.barycenter_net
l=list(np.sort(gpc_p.scores[:,n_gpc]))

percentiles=list()
percentiles.append(np.percentile(l, 1))
percentiles.append(np.percentile(l, 25))
percentiles.append(np.percentile(l, 50))
percentiles.append(np.percentile(l, 75))
percentiles.append(np.percentile(l, 90))

G_along_GPC2=GraphSet()
for i in range(len(percentiles)):
    G_along_GPC2.add(gpc_p.add(1,Bar,percentiles[i],Vector,range(Vector.n_nodes)))

plt.figure()
norm = mpl.colors.Normalize(vmin=-120, vmax=120, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='seismic')

for i in range(len(G_along_GPC2.X)):
    G_plot1=G_along_GPC2.X[i].to_networkX(layer=0, node_too=True)
    #pos=nx.kamada_kawai_layout(G_plot1)
    colors=list(nx.get_edge_attributes(G_plot1,'weight').values())
    colors_nodes=list(nx.get_node_attributes(G_plot1,'weight').values())
    to_rgb_col=[mapper.to_rgba(i) for i in colors]
    to_rgb_col_nodes=[mapper.to_rgba(i) for i in colors_nodes]
    # Draw the network
    nx.draw(G_plot1, pos, node_color=to_rgb_col_nodes, edge_color=to_rgb_col, width=4,#,norm=norm,
           with_labels=False)
    plt.show()


n_gpc=1
Vector=gpc_p.e_vec.X[n_gpc]
Bar=gpc_p.barycenter_net
l=list(np.sort(gpc_p.scores[:,n_gpc]))

percentiles=list()
percentiles.append(np.percentile(l, 1))
percentiles.append(np.percentile(l, 25))
percentiles.append(np.percentile(l, 50))
percentiles.append(np.percentile(l, 75))
percentiles.append(np.percentile(l, 90))

G_along_GPC4=GraphSet()
for i in range(len(percentiles)):
    G_along_GPC4.add(gpc_p.add(1,Bar,percentiles[i],Vector,range(Vector.n_nodes)))

plt.figure()
norm = mpl.colors.Normalize(vmin=-120, vmax=120, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='seismic')

pos={0: [-1.,  -2.],
1: [-2., 0.        ],
4: [ 0., 2.],
3: [2. , 0.],
2: [1. , -2.] }


for i in range(len(G_along_GPC4.X)):
    G_plot1=G_along_GPC4.X[i].to_networkX(layer=0, node_too=True)
    #pos=nx.kamada_kawai_layout(G_plot1)
    colors=list(nx.get_edge_attributes(G_plot1,'weight').values())
    colors_nodes=list(nx.get_node_attributes(G_plot1,'weight').values())
    to_rgb_col=[mapper.to_rgba(i) for i in colors]
    to_rgb_col_nodes=[mapper.to_rgba(i) for i in colors_nodes]
    # Draw the network
    nx.draw(G_plot1, pos, node_color=to_rgb_col_nodes, edge_color=to_rgb_col, width=4,#,norm=norm,
           with_labels=False)
    plt.show()



for i in range(len(G_along_GPC1.X)):
   print(match1.dis(G_along_GPC1.X[i],G_along_GPC2.X[i]))
   print(match1dis(G_along_GPC3.X[i],G_along_GPC4.X[i]))

print("SCORES")
print(gpc_c.scores)
print(gpc_p.scores)

print("VAL")
print(gpc_c.e_val)
print(gpc_p.e_val)

print("BARICENTER")
print(gpc_c.barycenter)
print(gpc_p.barycenter)

print("BARICENTER")
print(gpc_c.barycenter_net.x)
print(gpc_p.barycenter_net.x)

print("VECS")
for i in range(len(gpc_c.e_vec.X)):
    print(gpc_c.e_vec.X[i].x)
print("\n")
for i in range(len(gpc_p.e_vec.X)):
   print(gpc_p.e_vec.X[i].x)
