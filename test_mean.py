from core import Graph
from core import GraphSet
import meanc
from AlignCompute import mean_aac
from matcher import BK, GA, ID
import os
import sys
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm

X = GraphSet()

#X.read_from_text("Pentagones_10_100_500_Perm.txt")
X.read_from_text("Mammals_Grooming_kickouts_less20.txt")

match1=GA()
match2=GA(c=True)
match3=ID()
muc = mean_aac(X, match2,cm=True)
mu = mean_aac(X, match1,cm=False)

start2=time.time()
muc.align_and_est(50)
MUc = muc.mean
end2=time.time()
print("Tempo Impiegato:"+str(end2-start2))

print(MUc.x)

start1=time.time()
mu.align_and_est(50)
MU = mu.mean
end1=time.time()
print("Tempo Impiegato:"+str(end1-start1))

print(MU.x)

print('The distance between the two means is:   '+str(match3.dis(MU,MUc)))

G_plot1=MUc.to_networkX(layer=0, node_too=True)
G_plot2=MU.to_networkX(layer=0,node_too=True)

#pos={0: [-1.,  -2.],
#2: [-2., 0.        ],
#1: [ 0., 2.],
#4: [2. , 0.],
#3: [1. , -2.] }


pos={0: [-4. ,0.],
1: [-1.66, 0.757],
2: [-3.4, 2.107],
3: [-1.224, 1.12],
4: [-1.8075, 3.5683],
5: [-0.2748, 2.543],
6: [0.,4.],
7: [0.2748, 2.543],
8: [1.8075, 3.5683],
9: [1.224, 1.12],
10: [3.4, 2.107],
11: [1.66, 0.757],
13: [1.66, -0.757],
12: [4.,0.],
14: [3.4, -2.107],
15: [1.224, -1.12],
16: [1.8075, -3.5683],
17: [0.2748, -2.543]
}


colors=list(nx.get_edge_attributes(G_plot1,'weight').values())
colors_nodes=list(nx.get_node_attributes(G_plot1,'weight').values())
normalized = [(c-100)/110 for c in colors]
plt.figure()
# normalize the colors
#norm = mpl.colors.Normalize(vmin=-120, vmax=120, clip=True)
norm = mpl.colors.Normalize(vmin=0.001, vmax=0.7, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
to_rgb_col=[mapper.to_rgba(i) for i in colors]
to_rgb_col_nodes=[mapper.to_rgba(i) for i in colors_nodes]

# Draw the network
nx.draw(G_plot1,pos=pos,node_color=to_rgb_col_nodes, edge_color=to_rgb_col, width=4,#,norm=norm,
           with_labels=False)

plt.show()


##plot of the 2nd graph
colors=list(nx.get_edge_attributes(G_plot2,'weight').values())
colors_nodes=list(nx.get_node_attributes(G_plot2,'weight').values())
normalized = [(c-100)/110 for c in colors]
plt.figure()
## normalize the colors
#norm = mpl.colors.Normalize(vmin=-120, vmax=120, clip=True)
norm = mpl.colors.Normalize(vmin=0.001, vmax=0.7, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
to_rgb_col=[mapper.to_rgba(i) for i in colors]
to_rgb_col_nodes=[mapper.to_rgba(i) for i in colors_nodes]

## Draw the network
nx.draw(G_plot2, pos=pos, node_color=to_rgb_col_nodes, edge_color=to_rgb_col, width=4,#,norm=norm,
           with_labels=False)
plt.show()
