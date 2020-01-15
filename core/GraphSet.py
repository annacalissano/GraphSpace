import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix,vstack
import copy
from core import Graph
# Create a list of Graph

# Jain is creating a deep copy here two


class GraphSet:
    
    def __init__(self,graph_type=None):
        self.X = []
        if(graph_type==None):
            self.g_type='undirected'
        else:
            self.g_type=str(graph_type)
        self.y='label'
        self.node_attr='number'
        self.edge_attr='number'
        # if(len(set_graphs)!=0):
        #    self.X=[set_graphs[i] for i in range(len(set_graphs))]

    # Add a new Graph or initialize a new set
    def add(self,x):
        #if len(self.X)==0:
        #    self.X[0]=x
        #else:
        self.X.append(x)
    
    # Select Subset
    def sublist(self,index_set):
        if(min(index_set)<0 or max(index_set)<=min(index_set) or len(self.X)<max(index_set)):
            print("Hi! Give me a correct interval so I can sublist your list.")
            return 0
        else:
            X_new=GraphSet()
            for i in index_set:
                X_new.add(self.X[i])
            return X_new

    # ClassLabel: get the class label of a graph
    def ClassLabel(self,i):
        if(i<0 or len(self.X)<=i):
            print("Hi! Give me a correct index of graph so I can tell you its label.")
        else:
            return self.X[i].ClassLabel()
      
    # ClassLabel: get the class label of a graph
    def OutputLabel(self,i):
        if(i<0 or len(self.X)<=i):
            print("Hi! Give me a correct index of graph so I can tell you its label.")
        else:
            return self.X[i].OutputLabel()
        
    # Create a deep copy of the graphs set
    def cp(self):
        X=copy.deepcopy(self)
        return X
    # Conversion function: in Jain Code there are two function: toArray and toString changing the format of the set
    # I am not getting the meaning of it. Stay tuned, we will see.
    
    
    # Size of the dataset
    def size(self):
        return len(self.X)
    
    # Numero of nodes in the dataset
    def nodes_networks(self):
        self.n_nodes=max([self.X[i].nodes() for i in range(self.size())])
        
    # Gives back a set of graphs grown to be all of the same size       
    def grow_to_same_size(self):
        G2=self.cp()
        G2.nodes_networks()
        for i in range(G2.size()):
            if(G2.X[i].nodes()<G2.n_nodes):
                G2.X[i].grow(G2.n_nodes)
        return G2
    
    # Nodes Attributes:
    def get_node_attr(self):
        self.node_attr=max([self.X[i].node_attr for i in range(self.size())])
    # Edge Attributes
    def get_edge_attr(self):
        self.edge_attr=max([self.X[i].edge_attr for i in range(self.size())])
    
    # Call the graph function feature scale for all the element in the graphset
    def feature_scale(self):
        for i in range(self.size()):
            self.X[i].feature_scale()

    # Write to text file a GraphSet
    def write_to_text(self,filename):
        fh = open(filename,"w")
        i=0

        fh.writelines("GraphSet"+" "+str(self.size())+'\n')
        fh.writelines("GRAPH_TYPE" + " " + str(self.g_type) + '\n')
        i=next(k for k in range(self.size()) if self.X[k].edge_attr!=0)
        if(self.X[i].node_attr>=2):
            n_at=type(self.X[i].x[self.X[i].nodes_list()[0]][0]).__name__
        else:
            #n_at='list'
            n_at=type(self.X[i].x[self.X[i].nodes_list()[0]]).__name__
        if(self.X[i].edge_attr>=2):
            e_at=type(self.X[i].x[self.X[i].edges_list()[0]][0]).__name__
        elif(self.X[i].edge_attr>0):
            e_at=type(self.X[i].x[self.X[i].edges_list()[0]]).__name__
        fh.writelines("NODE_ATTR"+' '+' '.join([n_at for i in range(self.X[i].node_attr)])+'\n')
        fh.writelines("EDGE_ATTR"+' '+' '.join([e_at for i in range(self.X[i].edge_attr)])+'\n')
        fh.writelines("LABELS"+' '+type(self.X[0].y).__name__+'\n')
        
        for i in range(self.size()):
            n_attr=len(self.X[i].x)
            n_nodes=len(self.X[i].adj)
            fh.writelines("Graph"+" "+str(i)+" "+str(n_attr+n_nodes+2)+" "+'Label'+' '+str(self.X[i].y)+'\n') 
            fh.writelines("Attributes Dictionary"+" "+str(n_attr)+'\n') 
            for k,v in self.X[i].x.items():
                fh.write(" ".join(str(x) for x in k)+" "+" ".join(str(x) for x in v)+'\n')
            fh.writelines("Adjency List"+" "+str(n_nodes)+'\n') 
            for k,v in self.X[i].adj.items():
                fh.write(str(k)+" "+" ".join(str(x) for x in v)+'\n')
            #fh.write('End'+'\n')
            #fh.write('\n')
    
        fh.close()
    
    # Read from file
    def read_from_text(self,filename):
        fh = open(filename,"r")
        # dimension of the graphset
        n=int(fh.readline().split()[1])
        i=1
        e=enumerate(fh)
        
        #e.next()
        # Parse all the graphs
        while(n>0):
            #g=fh.readline().split()
            g=e.next()[1].split()
            print(g)
            dimG=int(g[2])
            #print dimG
            y=g[4]
            #y=None
            # Parse one graph
            while(dimG>0):
                dimAttr=int(e.next()[1].split()[2])
                x={}
                dimG-=1
                # Parse the attribute list
                while(dimAttr>0):
                    l=e.next()[1].split()
                    x[int(l[0]),int(l[1])]=map(lambda s: s if s.isalpha() else float(s), l[2:])
                    dimAttr-=1
                    dimG-=1
                dimEdg=int(e.next()[1].split()[2])
                adj={}
                dimG-=1
                # Parse the adj list
                while(dimEdg>0):
                    l=e.next()[1].split()
                        #print l
                    if(l[1:]>0):
                        adj[int(l[0])]=map(int, l[1:])
                    else:
                        adj[int(l[0])]=[]  
                    dimEdg-=1
                    dimG-=1
                if(y=='None'):
                    self.add(Graph(x=x,adj=adj,y=None))
                else:
                    self.add(Graph(x=x,adj=adj,y=y))
                del x
                del adj
                dimG-=1
            
            n-=1
    
    
    # Read tgf files
    def read_from_text2(self,filename):
        fh = open(filename,"r")
        for l in fh:
            n=0
            #process(l)
            g = l.split()
            
            if not g:
                continue
            #g=l.split()
            else:
                if(g[0]=='GraphSet'):
                    print('Start Parsing')
                    continue
                if(g[0]=='GRAPH_TYPE'):
                    graph_type=g[1]
                    self.g_type=graph_type
                    continue
                if(g[0]=='LABELS'):
                    type_y=g[1]
                    continue
                if(g[0]=='NODE_ATTR'):
                    n_attr=len(g)-1
                    continue
                if(g[0]=='EDGE_ATTR'):
                    e_attr=len(g)-1
                    
                    continue
                if(g[0]=='Graph'):
                    
                    if(int(g[1])==0):
                        x={}
                        # first graph need to be estimated
                        y=g[4]
                        continue
                    elif(int(g[1])>0):
                        # estimate
                        self.add(Graph(x=x,adj=adj,y=y))
                        x={}
                        y=g[4]
                        continue
                if(g[0]=='Attributes'):
                    block='attr'
                    continue
                if(g[0]=='Adjency'):
                    adj={}
                    block='adj'
                    continue
                elif(isinstance(int(g[0]),int)):
                    if(block=='attr'):
                        if(int(g[0])==int(g[1]) and n_attr>1):
                            x[int(g[0]),int(g[1])]=list(map(float,g[2:n_attr+2]))
                            continue
                        if(int(g[0])==int(g[1]) and n_attr==1):
                            x[int(g[0]),int(g[1])]=[float(g[2])]
                            
                        if(int(g[0])!=int(g[1]) and e_attr>1):
                            x[int(g[0]),int(g[1])]=list(map(float,g[2:e_attr+2]))
                            if(graph_type=='undirected'):
                                x[int(g[1]),int(g[0])]=x[int(g[0]),int(g[1])]
                            continue
                        if(int(g[0])!=int(g[1]) and e_attr==1):
                            x[int(g[0]),int(g[1])]=[float(g[2])]
                            if (graph_type == 'undirected'):
                                x[int(g[1]),int(g[0])]=x[int(g[0]),int(g[1])]
                            continue
                        continue
                    if(block=='adj'):
                        adj[int(g[0])]=list(map(int,g[0:len(g)]))
                        continue
                else: continue
        self.add(Graph(x=x,adj=adj,y=y))
        print("End Parsing")

    
    
    # Read tgf files
    def read_from_tgf(self,filename):
        fh = open(filename,"r")
        for l in fh:
            n=0
            #process(l)
            g = l.split()
            if not g:
                continue
            #g=l.split()
            else:
                if(g[0]=='GRAPH_TYPE' and g[1]=='undirected'):
                    g_type="undirected"
                    continue
                elif(g[0]=='GRAPH_TYPE' and g[1]=='directed'):
                    g_type="directed"
                    continue

                if(g[0]=='NODE_ATTR'):
                    n_attr=len(g)-1
                    print(n_attr)
                    continue
                if(g[0]=='EDGE_ATTR'):
                    e_attr=len(g)-1
                    print(e_attr)
                    continue
                if(g[0]=='GRAPH'):
                    x={}
                    continue
                if(g[0]=='NODES'):
                    block='n'
                    continue
                if(g[0]=='EDGES'):
                    block='e'
                    continue
                if(g[0]=='LABEL'):
                    y=g[1]
                if(g[0]=='#'):
                    self.add(Graph(x=x,adj=None,y=y))
                    del x,block
                    continue
                else:
                    try:
                        if(isinstance(int(g[0]),int)):

                            if(block=='n'):
                                x[int(g[0]),int(g[0])]=list(map(float,g[1:n_attr+1]))
                                continue
                            if(block=='e' and g_type=='undirected' and e_attr==1):
                                x[int(g[0]),int(g[1])]=[int(g[2])]
                                x[int(g[1]),int(g[0])]=[int(g[2])]
                                
                                continue
                            if(block=='e' and g_type=='undirected' and e_attr>1):
                                x[int(g[0]),int(g[1])]=list(map(float,g[2:e_attr+1]))
                                x[int(g[1]),int(g[0])]=list(map(float,g[2:e_attr+1]))
                                continue
                            if(block=='e' and g_type=='directed'):
                                x[int(g[0]),int(g[1])]=list(map(float,g[2:e_attr+1]))
                                continue
                    except:
                        continue

    def to_matrix_with_attr(self):
        # Number of Graphs
        n=self.size()
        self.nodes_networks()
        N=self.n_nodes
        self.get_node_attr()
        self.get_edge_attr()
        n_a=self.node_attr
        e_a=self.edge_attr
        # Column are all possible nodes and edges with attributes
        col = [str(item) for sublist in [[(i_r,i_c)]*n_a if i_r==i_c else [(i_r,i_c)]*e_a for i_r in range(N) for i_c in range(N)] for item in sublist]
        columns=list(map(lambda x: x[1] + str(col[:x[0]].count(x[1]) + 1) if col.count(x[1]) > 1 else x[1], enumerate(col)))
        # creating the empty dataframe
        D=pd.DataFrame(columns=columns)
        for i in range(self.size()):
            # Every network is added as a row
            col_i=[str(item) for sublist in [[k]*n_a if k[0]==k[1] else [k]*e_a for k in self.X[i].x.keys()] for item in sublist]
            col_i2=list(map(lambda x: x[1] + str(col_i[:x[0]].count(x[1]) + 1) if col_i.count(x[1]) > 1 else x[1], enumerate(col_i)))
            df_0 = pd.DataFrame([np.array([float(item) for sublist in [v for v in self.X[i].x.values()] for item in sublist])],columns=col_i2)
            D = pd.concat([D, df_0], axis=0,sort=False,ignore_index=True)
            del col_i,col_i2,df_0
        M=D.fillna(0)
        return M

    def drop_nodes(self,id):
        G_drop=GraphSet()
        for n in range(self.size()):
            G_drop.add(self.X[n].drop_nodes(id))
        return G_drop