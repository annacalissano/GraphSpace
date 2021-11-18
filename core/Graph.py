import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, vstack
import copy
import sys
import networkx as nx


# Class Graph
# Create a graph element, which is the base element for the usage of this package
# Graphs can be: undirect, direct, with scalar or vector euclidean attribute, with nodes or edges attributes of same or different dimension
# A Graph object is made of:
# - Adj: dictionary showing the structure (every node is a key, the values are the adjency nodes to the key)
# - x: dictionary showing the attributes
#   x[i,i]: nodes attributes
#   x[i,j]: edge attributes 
# - y: scalar or category attribute of the Graph (seful for classification or regression)

class Graph:

    # Initializing the graph
    # Input:
    # x: a dictionary with nodes attributes in [i,i] key and edge attributes in [j,i] position
    # Facultative Input:
    # adj: adjency list with [i] node as [k,l,m] the nodes is linked to as values (if None it is created in definethegraph())
    # y: numeric or categorical variable associated to the
    def __init__(self, x, adj, s):
        if (s is None):
            self.s = None
        else:
            self.s = s  # Regressor or class of the network
        # Check on the x: the null nodes should not be in the middle.
        # e.g. [0,1,2] can not have (0,0) (2,2) different from zero and (1,1) zero.
        self.x = copy.deepcopy(x)  # nodes and edges attributes
        self.attr = None  # matrix of feature vectors
        if (adj is None):
            self.adj = None
        else:
            self.adj = adj;  # adjacency list
        self.n_nodes = 0;  # number of nodes
        self.n_edges = 0;  # number of edges
        self.definethegraph()

    # Building the adj from x
    def definethegraph(self):
        # Function to define the adjency List

        # if it is empty, there is nothing to fill
        if (self.x == None):
            self.adj = None
            self.n_edges = 0
            self.n_nodes = 0
            self.node_attr = 0
            self.edge_attr = 0
        else:
            # number of nodes, edges and dimension of the structure
            # number of edges: (initialized and updated after)
            self.n_edges = 0
            # number of nodes:
            self.n_nodes = max([int(s[0]) if (s[0] >= s[1]) else s[1] for s in self.x.keys()]) + 1
            # length of node attributes
            _na = [len(v) for k, v in self.x.items() if type(v) is not int and k[0] == k[1]]
            if (not _na):
                self.node_attr = 1
            else:

                self.node_attr = max(_na)
                if (self.node_attr == 0):
                    self.node_attr = 1

            # length of edge attributes
            _ea = [len(v) for k, v in self.x.items() if type(v) is not int and k[0] != k[1]]
            if (not _ea):
                self.edge_attr = 0
            else:
                self.edge_attr = max(_ea)
                if (self.edge_attr == 0):
                    self.edge_attr = 1
            # building the adjency list:
            if (self.adj == None):
                self.adj = {}
                for i in range(self.n_nodes):
                    # All the nodes have an attribute by default
                    if (not (i, i) in self.x):
                        self.x.update({(i, i): [0] * self.node_attr})
                    # list of adjency nodes to node i
                    s = []
                    # self.x is a dictionary structure
                    for k, v in self.x.items():
                        # adjency nodes:
                        if (k[1] != k[0] and k[0] == i):
                            s = s + [k[1]]
                            # if(k[1] == i): s=s+[k[0]]
                    s = list(set(s))
                    if (len(s) > 0):
                        self.adj[i] = s
                        self.n_edges = self.n_edges + len(self.adj[i])
                    # else:
                    #    self.adj[i]=[]
                    # creating the self.x matrix (i.e. the attribute dictionary)
                    for j in range(self.n_nodes):
                        if ((i, j) in self.x):
                            if (type(self.x[i, j]) is not int):
                                _l = len(self.x[i, j])
                            else:
                                _l = 1
                            if (i == j and _l < self.node_attr and _l > 1): self.x[i, j] = self.x[i, j] + [0] * (
                                        self.node_attr - _l)
                            if (i == j and _l < self.node_attr and _l == 1 and isinstance(self.x[i, j], list)):
                                self.x[i, j] = self.x[i, j] + [0] * (self.node_attr - _l)
                            if (i == j and _l < self.node_attr and _l == 1 and not isinstance(self.x[i, j], list)):
                                self.x[i, j] = [self.x[i, j]] + [0] * (self.node_attr - _l)
                            if (i != j and _l < self.edge_attr and _l > 1): self.x[i, j] = self.x[i, j] + [0] * (
                                        self.edge_attr - _l)
                            if (i != j and _l < self.edge_attr and _l == 1 and isinstance(self.x[i, j], list)): self.x[
                                i, j] = self.x[i, j] + [0] * (self.edge_attr - _l)
                            if (i != j and _l < self.edge_attr and _l == 1 and not isinstance(self.x[i, j], list)):
                            self.x[i, j] = [self.x[i, j]] + [0] * (self.edge_attr - _l)
                        # else:
                        #    if(i==j): self.x[i,j]=[0]*self.node_attr
                        #    else: self.x[i,j]=[0]*self.edge_attr

    # If we are intetrested in the attributes, uncomment what is after the 'and'
    def __eq__(self, other):
        # """Override the default Equals behavior"""
        return self.x == other.x  # and self.attr == other.attr

    def __ne__(self, other):
        # """Override the default Unequal behavior"""
        return self.x != other.x  # or self.attr != other.attr

    # Create a deep copy of the object
    def cp(self):
        d = copy.deepcopy(self)
        return d

    # Number of nodes
    def nodes(self):
        return self.n_nodes

    # Number of edges
    def edges(self):
        return self.n_edges

    # Degree of node (unweighted)
    def degree(self, i):
        if (i < 0 or self.n_nodes <= i or not i in self.adj):
            return 0;
        # print i
        else:
            return len(self.adj[i])

    # Degree of node (weighted)
    def weighted_degree(self, i):
        if (i < 0 or self.n_nodes <= i or not i in self.adj):
            return 0;
        return sum(self.adj[i])

    # Adjency dictionary with attributes
    def matrix(self):
        return self.x

    # Adjency List
    def adjList(self):
        return self.adj

    # The following three functions deal with the nodel class label if there is one:
    def Features(self):
        if (self.s is None):
            print("Missing attribute s.")
        else:
            return s

    # Class Label assignment
    def setFeatures(self, feature):
        self.s = feature

    def OutputFeature(self):
        if (self.s is None):
            print("Missing features.")
        else:
            return self.s

    # Check if it has a label
    def HasFeatures(self):
        if (self.s is None):
            return False
        else:
            return True

    # Attributes node dimension
    def dimNodes(self):
        for i in range(self.n_nodes):
            if ((i, i) in self.x and self.x[i, i] != 0):
                return len(self.x[i, i])
        return 0

    # Attributes edge dimension
    def dimEdges(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if ((i, j) in self.x and self.x[i, j] != 0):
                    return len(self.x[i, j])
        return 0

    # Return keys of nodes
    def nodes_list(self):
        n_list = []
        for k, v in self.x.items():
            if (k[0] == k[1]):
                n_list.append(k)
        return n_list

    # Return keys of edges
    def edges_list(self):
        e_list = []
        for k, v in self.x.items():
            if (k[0] != k[1]):
                e_list.append(k)
        return e_list

    # Get the adjency sparse matrix of 0 and 1
    def get_pure_matrix(self):
        _x = lil_matrix((self.n_nodes, self.n_nodes))
        for i in self.x.keys():
            _x[i] = 1
        return _x

    # Check if the graph is empty:
    def isZero(self):
        for i in range(self.n_nodes):
            if (self.x[i, i] != 0):
                return False;
            degree = self.degree(i)
            for j in range(degree):
                j0 = self.adj[i][j]
                if (self.x[i][j0] != 0):
                    return False
                return True

    # Scale: multuply all the weights (nodes and edges) with 'a'
    def scale(self, a):
        x_new = {}
        for i in range(self.n_nodes):
            x_new[i, i] = list(np.multiply(a, self.x[i, i]))
            degree = self.degree(i)
            for j in range(degree):
                j0 = self.adj[i][j]
                x_new[i, j0] = list(np.multiply(a, self.x[i, j0]))
        return Graph(x=x_new, adj=self.adj, s=self.s)

    # Permuting indexes of the nodes
    def permutelist(self):
        f = np.random.permutation(self.nodes())
        return f

    # Permute function
    # This function is a key function when dealing with more than one network.
    # it is permuting the node of the network and it is called in all the alignment process
    # Input:
    # - f: list of index of nodes (e.g. of a permutation of 3 nodes network: f=[1,0,2])
    def permute(self, f):
        _x = {}
        _adj = {}
        for i in range(self.n_nodes):
            fi = f[i]
            _x[fi, fi] = self.x[i, i]
            _adj[fi] = []
            for j in range(self.degree(i)):
                j0 = self.adj[i][j]
                fj = f[j0]
                _x[f[i], f[j0]] = self.x[i, j0]
                _adj[fi].append(fj)
        del (self.x, self.adj)
        self.x = copy.deepcopy(_x)
        self.adj = copy.deepcopy(_adj)

    # Scale the attributes of node and edges within a network
    def feature_scale(self):
        # initialize both minimum and maximum for attributes
        x_M_nodes = [None] * self.node_attr
        x_m_nodes = [sys.maxint] * self.node_attr
        x_M_edges = [None] * self.edge_attr
        x_m_edges = [sys.maxint] * self.edge_attr
        _x = self.x
        for i in range(self.n_nodes):

            x_M_nodes = np.maximum(x_M_nodes, self.x[i, i])
            x_m_nodes = np.minimum(x_m_nodes, self.x[i, i])

            for j in range(self.degree(i)):
                j0 = self.adj[i][j]
                x_M_edges = np.maximum(x_M_edges, self.x[i, j0])
                x_m_edges = np.minimum(x_m_edges, self.x[i, j0])
        range_nodes = np.array(np.subtract(x_M_nodes, x_m_nodes), dtype=float)
        range_edges = np.array(np.subtract(x_M_edges, x_m_edges), dtype=float)
        for i in range(self.n_nodes):
            a = np.array(np.subtract(_x[i, i], x_m_nodes), dtype=float)
            _x[i, i] = np.divide(a, range_nodes, out=np.zeros_like(a), where=range_nodes != 0).tolist()
            for j in range(self.degree(i)):
                j0 = self.adj[i][j]
                if (all(range_edges == 0)):
                    _x[i, j0] = np.divide(_x[i, j0], _x[i, j0]).tolist()
                else:
                    a = np.array(np.subtract(_x[i, j0], x_m_edges), dtype=float)
                    _x[i, j0] = np.divide(a, range_edges, out=np.zeros_like(a), where=range_edges != 0).tolist()
        self.x = _x

    # Increasing graph size adding new empty nodes to the network
    # input:
    # - size: scalar value representing the new desired size
    def grow(self, size):
        if (size <= self.n_nodes):
            return self
        else:
            if (self.attr is None):
                _x = {}
                _adj = {}
                for i in range(size):
                    for j in range(size):

                        if (i <= (self.n_nodes - 1) and j <= (self.n_nodes - 1)):
                            if ((i, j) in self.x):
                                # _adj[i]=self.adj[i]
                                _x[i, j] = self.x[i, j]

                        else:
                            if (i == j):
                                _x[i, j] = [0] * self.node_attr
                                # _adj[i]=[]
                            # else:
                            #    _x[i,j]=[0]*self.edge_attr

                self.x = _x
                # self.adj=_adj
                self.n_nodes = size
                # self.definethegraph()
            else:
                print("Hi Darling, attributes needed: use function grow_with_attributes instead!")

    # Increasing graph size adding new nodes to the network with the given attributes
    # input:
    # - size: scalar value representing the new desired size
    # - new_attr: attribute to be added
    def grow_with_attributes(self, size, new_attr):
        if (size <= self.n_nodes):
            return self
        else:
            _x = {}
            _adj = {}
            for i in range(size):
                if (i <= (self.n_nodes - 1)):
                    _adj[i] = self.adj[i]
                    for j in range(self.n_nodes):
                        _x[i, j] = self.x[i, j]
                else:
                    _adj[i] = []
            self.x = _x
            self.adj = _adj
            self.n_nodes = size
            self.attr = self.attr.append(new_attr)

    # Decrising the graph size
    # input:
    # - size: scalar value representing the new desired size
    def shrink(self, size):
        if (size < 1 or self.n_nodes <= size):
            print("Invalid size")
        else:
            if (size < self.n_nodes):
                _x = {}
                for i in range(size):
                    for j in range(size):
                        _x[i, j] = self.x[i, j]
                self.x = _x
                self.adj = Graph(self.x, y=None, adj=None)
                if (self.attr is not None):
                    self.attr = self.attr.loc[range(size), :]

    # Extract the network corresponding to the j-th edge layer
    # input:
    # - j: the desired layer
    # - node_too: if True extract also the layer from the node
    def extract_layer(self, j, node_too):
        # j is the j-th edges' attribute to built the network on
        _x = copy.deepcopy(self.x)
        for k, v in _x.items():
            if (node_too == True):
                _x[k] = [v[j]]
            else:
                if (k[0] != k[1]):
                    _x[k] = [v[j]]
        return _x

    # Delete the attribute selected:
    # input:
    # - j: the attribute to delete
    # - attr_type: either node or edge according to which
    def del_attribute(self, j, attr_type):
        if (attr_type == 'node'):
            if (self.dimNodes() < j):
                return "Error: the attribute index is too high"
            else:
                for k, v in self.x.items():
                    if (k[0] == k[1]):
                        del v[j]

    # From Graph to vector structure (1 if there is a link, 0 otherwise)
    # The vector is building unrolling the adj matrix by row
    # The dimension of the vector is #edges*n_attr_edges+#nodes*n_attr_nodes= N*(N-1)*#e_attr+N*n_attr
    def to_vector_with_attributes(self):
        n_a = self.node_attr
        e_a = self.edge_attr
        col_i = [str(item) for sublist in [[k] * n_a if k[0] == k[1] else [k] * e_a for k in self.x.keys()] for item in
                 sublist]
        col_i2 = list(map(lambda x: x[1] + str(col_i[:x[0]].count(x[1]) + 1) if col_i.count(x[1]) > 1 else x[1],
                          enumerate(col_i)))
        df_0 = pd.DataFrame([np.array([item for sublist in [v for v in self.x.values()] for item in sublist])],
                            columns=col_i2)
        return df_0

    # INTERMEDIATE STEP: # ATTRIBUTES == 1
    # From Graph to vector structure, for all the elements in the set iset of indices (i,j) given
    # If the graph has no (i,j) element, its value is set to 0
    # The vector is building unrolling the adj matrix by row
    # The dimension of the vector is len(iset)*n_attr_edges or len(iset)*n_attr_nodes
    def to_vector_with_select_attributes(self, iset):
        n_a = self.node_attr
        e_a = self.edge_attr
        assert n_a * e_a == 1, "This method assumes that the node or edge attribute is a scalar"
        # working on the column names:
        col_i = [str(item) for item in iset]
        # and then creating the dataframe
        df_0 = pd.DataFrame([np.array([self.x[item][0] if item in self.x.keys() else 0.0 for item in iset])],
                            columns=col_i)
        return df_0

    # GENERALIZATION to #attrib > 1 !!!
    # From Graph to vector structure, for all the elements in the set iset of indices (i,j) given
    # If the graph has no (i,j) element, its value is set to 0
    # The vector is building unrolling the adj matrix by row
    # The dimension of the vector is respectively #edges*n_attr_edges and #nodes*n_attr_nodes
    def to_vector_with_select_nodes(self, iset):
        n_a = self.node_attr
        # working on the column names:
        col_i = [str(item) for item in iset]
        # and then creating the dataFrame (actually no need to create an intermediate np.array)
        df_0 = pd.DataFrame([self.x[item] if item in self.x.keys() else [0.0] * n_a for item in iset], index=col_i)
        return df_0

    def to_vector_with_select_edges(self, iset):
        e_a = self.edge_attr
        # working on the column names:
        col_i = [str(item) for item in iset]
        # and then creating the dataFrame (actually no need to create an intermediate np.array)
        df_0 = pd.DataFrame([self.x[item] if item in self.x.keys() else [0.0] * e_a for item in iset], index=col_i)
        return df_0

    # To Networkx Object: this function convert the graph to a networkx object
    # input:
    # - layer: the layer to extract
    # - node_too: True if the layer should be extracted from the nodes, False otherwise
    def to_networkX(self, layer, node_too, directed):
        if (directed == True):
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        if (layer != None):
            _x = self.extract_layer(layer, node_too)
            G.add_nodes_from(range(self.n_nodes))
            G.add_edges_from([e for e in list(_x.keys()) if e[0] != e[1]])
            if (node_too == True):
                nx.set_node_attributes(G, dict((k[0], {'weight': float(v[0])}) for k, v in _x.items() if k[0] == k[1]))
            nx.set_edge_attributes(G, dict((k, float(v[0])) for k, v in _x.items()), 'weight')
        else:
            print("Caution: No Layer Specified")
            G.add_nodes_from(range(self.n_nodes))
            G.add_edges_from([e for e in list(_x.keys()) if e[0] != e[1]])
            if (node_too == True):
                nx.set_node_attributes(G, dict((k[0], {'weight': float(v[0])}) for k, v in _x.items() if k[0] == k[1]))
            nx.set_edge_attributes(G, self.x, 'weight')
        return G

    # Drop a node or a set of nodes
    # input:
    # - id: the index of the node
    def drop_nodes(self, id):
        adj_new = copy.deepcopy(self.adj)
        x_new = copy.deepcopy(self.x)
        for i in range(self.n_nodes):
            if (i in id):
                if ((i, i) in x_new):
                    del x_new[i, i]
                # all the edges exiting the node i
                if (i in adj_new):

                    current_id = [(i, j) for j in adj_new[i]]
                    print('Delete i:')
                    print(current_id)
                    for key in current_id: del x_new[key]
                    del adj_new[i], current_id
            else:
                # all the edges entering in node i
                current_id = [(i, j) for j in adj_new[i] if j in id]
                adj_new[i] = [n for n in adj_new[i] if n not in id]
                for key in current_id: del x_new[key]
        return Graph(x=x_new, adj=adj_new, y=self.s)
