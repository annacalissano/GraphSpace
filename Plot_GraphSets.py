import os
import sys
sys.path.append("C:\\Users\\Anna\\OneDrive - Politecnico di Milano\\Windows\\Polimi\\Ricerca\\Regression\\GraphSpace\\")
os.chdir('C:\\Users\\Anna\\OneDrive - Politecnico di Milano\\Windows\\Polimi\\Ricerca\\Regression\\Simulations\\DataSets')
from core import Graph
from core import GraphSet
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
# Modello regression
from sklearn import linear_model,gaussian_process
import numpy as np
import random
random.seed(3)
G=GraphSet(graph_type='undirected')
G.read_from_text("C:\\Users\\Anna\\OneDrive - Politecnico di Milano\\Windows\\Polimi\\Ricerca\\Regression\\Simulations\\DataSets\\GraphSet_CryptCorrMats.txt")


# plot the true and the predicted
G_origin=r.y_net_pred.X[0]
# Network plot
# Go to networkx format
G_plot=G_origin.to_networkX(layer=0,node_too=True)
# Define the nodes positions
pos={0: [-0.16210871,  0.92931688],
1: [0.36616978, 1.        ],
2: [ 0.48415449, -0.48927315]}
# or with networkx.layout https://networkx.github.io/documentation/stable/reference/drawing.html
# Initialize the colors as egdes weights
colors=list(nx.get_edge_attributes(G_plot,'weight').values())
#normalized = [(c-100)/110 for c in colors]
plt.figure()
# normalize the colors
norm = mpl.colors.Normalize(vmin=0, vmax=2, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
to_rgb_col=[mapper.to_rgba(i) for i in colors]
# Draw the network
nx.draw(G_plot, pos, node_color='#A0CBE2', edge_color=to_rgb_col, width=4,#,norm=norm,
            with_labels=False)
nx.draw_networkx_edge_labels(G_plot,pos)